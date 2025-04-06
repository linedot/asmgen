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
            $HEADER

            $VALIDFUN_DEF
            int main(int argc, char* argv[])
            {
                $TMPDEC

                $TMPASSIGN

                $INPARAMDEF
                $OUTPARAMDEF

                $BENCHPREPARE
                $DATAPREPARE

                for (std::uint64_t meas = 0; meas < measurements; meas++)
                {
                    $BENCHTIC
                    for (std::uint64_t iter = 0; iter < iterations; iter++)
                    {
                        __asm__ volatile (
                            $ASMBLOCK
                        );
                    }
                    $BENCHTOC
                }


                $BENCHRESULT


                if($VALIDATE)
                {
                    return 0;
                }
                else
                {
                    $ANALYZE
                    return -1;
                }
            }"""

BENCHMARK_SEPARATED = """
                $BENCHPREPARE
                $DATAPREPARE

                for (std::uint64_t iter = 0; iter < iterations; iter++)
                {
                    $BENCHTIC
                    for (std::uint64_t meas = 0; meas < measurements; meas++)
                    {
                        __asm__ volatile (
                            $ASMBLOCK
                        );
                    }
                    $BENCHTOC
                }
                $BENCHRESULT

                if($VALIDATE)
                {
                    benchmark_status[$BENCHNUM] = true
                }
                else
                {
                    $ANALYZE
                }
"""

BENCHMARK_MULTI = """
            $HEADER

            $VALIDFUN_DEF
            int main(int argc, char* argv[])
            {
                $TMPDEC

                std::array<std::string, $BENCHCOUNT> benchmark_names    = $BENCHNAMELIST;
                std::array<bool,        $BENCHCOUNT> benchmark_statuses = {false};

                $TMPASSIGN

                $INPARAMDEF
                $OUTPARAMDEF

                $BENCHMARKS


                for(std::size_t i = 0; i < $BENCHCOUNT; i++)
                {
                    std::cout << "Benchmark " << std::setw(30) << benchmark_names[i] << " (" << i << "/" << $BENCHCOUNT << "): "
                              << (benchmark_statuses[i] ? "Success" : "Failure") << "\n";
                }

            }"""
