# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
C++ templates for tests and benchmarks
"""
TEST_FUNC = """
            ${header}

            ${checkfun_definition}
            int ${function_name}(${function_params})
            {
                ${prepare}
                __asm__ volatile (
                    ${asmblock}
                );
                if(${check})
                {
                    return 0;
                }
                else
                {
                    ${analyze}
                    return -1;
                }
            }
"""
TEST_FUNC_DECLARATION = """
            int ${function_name}(${function_params});
"""
SIMPLE_FUNC = """
            void ${function_name}(${function_params})
            {
                ${prepare}
                __asm__ volatile (
                    ${asmblock}
                );
            }
"""

SIMPLE_TEST = """
           ${header}

           ${checkfun_definition}
           int main(int argc, char* argv[])
           {
               ${tmp_decl}

               ${tmp_assign}

               ${inparam_definition}
               ${outparam_definition}

               __asm__ volatile (
                   ${asmblock}
               );
               if(${check})
               {
                   return 0;
               }
               else
               {
                   ${analyze}
                   return -1;
               }
           }"""


BENCHMARK = """
            ${header}

            ${validfun_def}
            int main(int argc, char* argv[])
            {
                ${tmpdec}

                ${tmpassign}

                ${inparamdef}
                ${outparamdef}

                ${benchprepare}
                ${dataprepare}

                for (std::uint64_t meas = 0; meas < measurements; meas++)
                {
                    ${benchtic}
                    __asm__ volatile (
                        ${asmblock}
                    );
                    ${benchtoc}
                }


                ${benchresult}


                if(${validate})
                {
                    return 0;
                }
                else
                {
                    ${analyze}
                    return -1;
                }
            }"""

BENCHMARK_SEPARATED = """
                ${benchprepare}
                ${dataprepare}

                for (std::uint64_t iter = 0; iter < iterations; iter++)
                {
                    ${benchtic}
                    for (std::uint64_t meas = 0; meas < measurements; meas++)
                    {
                        __asm__ volatile (
                            ${asmblock}
                        );
                    }
                    ${benchtoc}
                }
                ${benchresult}

                if(${validate})
                {
                    benchmark_status[${benchnum}] = true
                }
                else
                {
                    ${analyze}
                }
"""

BENCHMARK_MULTI = """
            ${header}

            ${validfun_def}
            int main(int argc, char* argv[])
            {
                ${tmpdec}

                std::array<std::string, ${benchcount}> benchmark_names    = ${benchnamelist};
                std::array<bool,        ${benchcount}> benchmark_statuses = {false};

                ${tmpassign}

                ${inparamdef}
                ${outparamdef}

                ${benchmarks}


                for(std::size_t i = 0; i < ${benchcount}; i++)
                {
                    std::cout << "Benchmark " << std::setw(30) << benchmark_names[i] << " (" << i << "/" << ${benchcount} << "): "
                              << (benchmark_statuses[i] ? "Success" : "Failure") << "\n";
                }

            }"""
