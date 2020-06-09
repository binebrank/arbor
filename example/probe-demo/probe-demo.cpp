#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/simulation.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any.hpp>
#include <arbor/util/any_ptr.hpp>
#include <tinyopt/smolopt.h>

// Simulate a cell modelled as a simple cable with HH dynamics,
// emitting the results of a user specified probe over time.

using arb::util::any;
using arb::util::any_cast;
using arb::util::any_ptr;

const char* help_msg =
    "[OPTION]... PROBE\n"
    "\n"
    " --dt=TIME           set simulation dt to TIME [ms]\n"
    " --until=TIME        simulate until TIME [ms]\n"
    " -n, --n-cv=N        discretize with N CVs\n"
    " -t, --sample=TIME   take a sample every TIME [ms]\n"
    " -x, --at=X          take sample at relative position X along cable\n"
    " --exact             use exact time sampling\n"
    " -h, --help          print extended usage information and exit\n"
    "\n"
    "Simulate a simple 1 mm cable with HH dynamics, taking samples according\n"
    "to PROBE (see below). Unless otherwise specified, the simulation will\n"
    "use 10 CVs and run for 100 ms with a time step of 0.025 ms.\n"
    "\n"
    "Samples are by default taken every 1 ms; for probes that test a specific\n"
    "point on the cell, that point is 0.5 along the cable (i.e. 500 µm) unless\n"
    "specified with the --at option.\n"
    "\n"
    "PROBE is one of:\n"
    "\n"
    "    v           membrane potential [mV] at X\n"
    "    i_axial     axial (distal) current [nA] at X\n"
    "    j_ion       total ionic membrane current density [A/m²] at X\n"
    "    j_na        sodium ion membrane current density [A/m²] at X\n"
    "    j_k         potassium ion membrane current density [A/m²] at X\n"
    "    c_na        internal sodium concentration [mmol/L] at X\n"
    "    c_k         internal potassium concentration [mmol/L] at X\n"
    "    hh_m        HH state variable m at X\n"
    "    hh_h        HH state variable h at X\n"
    "    hh_n        HH state variable n at X\n"
    "\n"
    "where X is the relative position along the cable as described above, or else:\n"
    "\n"
    "    all_v       membrane potential [mV] in each CV\n"
    "    all_i_ion   total ionic membrane current [nA] in each CV\n"
    "    all_i_na    sodium ion membrane current [nA] in each CV\n"
    "    all_i_k     potassium ion membrane current [nA] in each CV\n"
    "    all_i       total membrane current [nA] in each CV\n"
    "    all_c_na    internal sodium concentration [mmol/L] in each CV\n"
    "    all_c_k     internal potassium concentration [mmol/L] in each CV\n"
    "    all_hh_m    HH state variable m in each CV\n"
    "    all_hh_h    HH state variable h in each CV\n"
    "    all_hh_n    HH state variable n in each CV\n";

struct options {
    double sim_end = 100.0;   // [ms]
    double sim_dt = 0.025;    // [ms]
    double sample_dt = 1.0;   // [ms]
    unsigned n_cv = 10;

    bool exact = false;
    bool scalar_probe = true;
    any probe_addr;
    std::string value_name;
};

const char* argv0 = "";
bool parse_options(options&, int& argc, char** argv);
void vector_sampler(arb::cell_member_type, arb::probe_tag, any_ptr, std::size_t, const arb::sample_record*);
void scalar_sampler(arb::cell_member_type, arb::probe_tag, any_ptr, std::size_t, const arb::sample_record*);

struct cable_recipe: public arb::recipe {
    arb::cable_cell_global_properties gprop;
    any probe_addr;

    explicit cable_recipe(any probe_addr, unsigned n_cv):
        probe_addr(std::move(probe_addr))
    {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        gprop.default_parameters.discretization = arb::cv_policy_fixed_per_branch(n_cv);
    }

    arb::cell_size_type num_cells() const override { return 1; }
    arb::cell_size_type num_probes(arb::cell_gid_type) const override { return 1; }
    arb::cell_size_type num_targets(arb::cell_gid_type) const override { return 0; }

    arb::probe_info get_probe(arb::cell_member_type probe_id) const override {
        return {probe_id, 0, probe_addr}; // (tag value 0 is not used)
    }

    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override {
        return arb::cell_kind::cable;
    }

    arb::util::any get_global_properties(arb::cell_kind) const override {
        return gprop;
    }

    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
        using namespace arb;
        const double length = 1000; // [µm]
        const double diam = 1; // [µm]

        sample_tree samples({msample{0, 0, 0, 0.5*diam}, msample{length, 0, 0, 0.5*diam}}, {mnpos, 0u});
        cable_cell c(samples);

        c.paint(reg::all(), "hh"); // HH mechanism over whole cell.
        c.place(mlocation{0, 0.}, i_clamp{0., INFINITY, 1.}); // Inject a 1 nA current indefinitely.
        return c;
    }
};

int main(int argc, char** argv) {
    argv0 = argv[0];
    try {
        options opt;
        if (!parse_options(opt, argc, argv)) {
            return 0;
        }

        cable_recipe R(opt.probe_addr, opt.n_cv);

        auto context = arb::make_context();
        arb::simulation sim(R, arb::partition_load_balance(R, context), context);

        sim.add_sampler(arb::all_probes,
                arb::regular_schedule(opt.sample_dt),
                opt.scalar_probe? scalar_sampler: vector_sampler,
                opt.exact? arb::sampling_policy::exact: arb::sampling_policy::lax);

        // CSV header for sample output:
        std::cout << "t, " << (opt.scalar_probe? "x, ": "x0, x1, ") << opt.value_name << '\n';

        sim.run(opt.sim_end, opt.sim_dt);
    }
    catch (to::option_error& e) {
        to::usage(argv0, "[OPTIONS]... PROBE\nTry '--help' for more information.", e.what());
        return 1;
    }
    catch (std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        return 2;
    }
}

void scalar_sampler(arb::cell_member_type, arb::probe_tag, any_ptr meta, std::size_t n, const arb::sample_record* samples) {
    auto* loc = any_cast<const arb::mlocation*>(meta);
    assert(loc);

    std::cout << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i<n; ++i) {
        auto* value = any_cast<const double*>(samples[i].data);
        assert(value);

        std::cout << samples[i].time << ", " << loc->pos << ", " << *value << '\n';
    }
}

void vector_sampler(arb::cell_member_type, arb::probe_tag, any_ptr meta, std::size_t n, const arb::sample_record* samples) {
    auto* cables_ptr = any_cast<const arb::mcable_list*>(meta);
    assert(cables_ptr);
    unsigned n_cable = cables_ptr->size();

    std::cout << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i<n; ++i) {
        auto* value_range = any_cast<const arb::cable_sample_range*>(samples[i].data);
        assert(value_range);
        assert(n_cable==value_range->second-value_range->first);

        for (unsigned j = 0; j<n_cable; ++j) {
            arb::mcable where = (*cables_ptr)[j];
            std::cout << samples[i].time << ", "
                      << where.prox_pos << ", "
                      << where.dist_pos << ", "
                      << value_range->first[j] << '\n';
        }
    }
}

bool parse_options(options& opt, int& argc, char** argv) {
    using std::get;
    using namespace to;

    auto do_help = [&]() { usage(argv0, help_msg); };

    // Map probe argument to output variable name, scalarity, and a lambda that makes specific probe address from a location.
    std::pair<const char*, std::tuple<const char*, bool, std::function<any (double)>>> probe_tbl[] {
        // located probes
        {"v",         {"v",       true,  [](double x) { return arb::cable_probe_membrane_voltage{{0, x}}; }}},
        {"i_axial",   {"i_axial", true,  [](double x) { return arb::cable_probe_axial_current{{0, x}}; }}},
        {"j_ion",     {"j_ion",   true,  [](double x) { return arb::cable_probe_total_ion_current_density{{0, x}}; }}},
        {"j_na",      {"j_na",    true,  [](double x) { return arb::cable_probe_ion_current_density{{0, x}, "na"}; }}},
        {"j_k",       {"j_k",     true,  [](double x) { return arb::cable_probe_ion_current_density{{0, x}, "k"}; }}},
        {"c_na",      {"c_na",    true,  [](double x) { return arb::cable_probe_ion_int_concentration{{0, x}, "na"}; }}},
        {"c_k",       {"c_k",     true,  [](double x) { return arb::cable_probe_ion_int_concentration{{0, x}, "k"}; }}},
        {"hh_m",      {"hh_m",    true,  [](double x) { return arb::cable_probe_density_state{{0, x}, "hh", "m"}; }}},
        {"hh_h",      {"hh_h",    true,  [](double x) { return arb::cable_probe_density_state{{0, x}, "hh", "h"}; }}},
        {"hh_n",      {"hh_n",    true,  [](double x) { return arb::cable_probe_density_state{{0, x}, "hh", "n"}; }}},
        // all-of-cell probes
        {"all_v",     {"v",       false, [](double)   { return arb::cable_probe_membrane_voltage_cell{}; }}},
        {"all_i_ion", {"i_ion",   false, [](double)   { return arb::cable_probe_total_ion_current_cell{}; }}},
        {"all_i_na",  {"i_na",    false, [](double)   { return arb::cable_probe_ion_current_cell{"na"}; }}},
        {"all_i_k",   {"i_k",     false, [](double)   { return arb::cable_probe_ion_current_cell{"k"}; }}},
        {"all_i",     {"i",       false, [](double)   { return arb::cable_probe_total_current_cell{}; }}},
        {"all_c_na",  {"c_na",    false, [](double)   { return arb::cable_probe_ion_int_concentration_cell{"na"}; }}},
        {"all_c_k",   {"c_k",     false, [](double)   { return arb::cable_probe_ion_int_concentration_cell{"k"}; }}},
        {"all_hh_m",  {"hh_m",    false, [](double)   { return arb::cable_probe_density_state_cell{"hh", "m"}; }}},
        {"all_hh_h",  {"hh_h",    false, [](double)   { return arb::cable_probe_density_state_cell{"hh", "h"}; }}},
        {"all_hh_n",  {"hh_n",    false, [](double)   { return arb::cable_probe_density_state_cell{"hh", "n"}; }}}
    };

    std::tuple<const char*, bool, std::function<any (double)>> probe_spec;
    double probe_pos = 0.5;

    to::option cli_opts[] = {
        { to::action(do_help), to::flag, to::exit, "-h", "--help" },
        { {probe_spec, to::keywords(probe_tbl)}, to::single },
        { opt.sim_dt,    "--dt" },
        { opt.sim_end,   "--until" },
        { opt.sample_dt, "-t", "--sample" },
        { probe_pos,     "-x", "--at" },
        { opt.n_cv,      "-n", "--n-cv" },
        { to::set(opt.exact), to::flag, "--exact" }
    };

    if (!to::run(cli_opts, argc, argv+1)) return false;
    if (!get<2>(probe_spec)) throw to::user_option_error("missing PROBE");
    if (argv[1]) throw to::user_option_error("unrecognized option");

    opt.value_name = get<0>(probe_spec);
    opt.scalar_probe = get<1>(probe_spec);
    opt.probe_addr = get<2>(probe_spec)(probe_pos);
    return true;
}

