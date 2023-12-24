from asmgen.cppgen.expressions import identity

def always_true(function_name, condition=""):
    return f"""
    bool {function_name}()
    {{
        return true;
    }}
    """

def always_false(function_name, condition=""):
    return f"""
    bool {function_name}()
    {{
        return false;
    }}
    """

def bi_u64_condition(function_name, condition):
    return f"""
    bool {function_name}(std::uint64_t a, std::uint64_t b)
    {{
        return {condition};
    }}
    """

def bi_ptr_condition(function_name, condition):
    return f"""
    bool {function_name}(std::uint64_t a, std::uint64_t b)
    {{
        return {condition};
    }}
    """

def bi_u64_relation(function_name, expression_generator, relation):
    return bi_u64_condition(function_name, f"b {relation} {expression_generator('a')}")

def bi_u64_eq(fn, eg):
    return bi_u64_relation(fn,eg, "==")

def bi_u64_leq(fn, eg):
    return bi_u64_relation(fn,eg, "<=")

def bi_u64_geq(fn, eg):
    return bi_u64_relation(fn,eg, ">=")

def bi_u64_le(fn, eg):
    return bi_u64_relation(fn,eg, "<")

def bi_u64_gr(fn, eg):
    return bi_u64_relation(fn,eg, ">")

# TODO: epsilon comparison? Random failures otherwise
def vec_fp32_condition(function_name, condition):
    return f"""
    bool {function_name}(const std::vector<float>& avec, const std::vector<float>& bvec)
    {{
        return std::equal(avec.begin(), avec.end(),
                          bvec.begin(),
                          [](float a, float b)
                          {{
                              return {condition};
                          }});
    }}
    """

def vec_fp32_relation(function_name, expression_generator, relation):
    return vec_fp32_condition(function_name, f"b {relation} {expression_generator('a')}")

def vec_fp32_eq(fn, eg):
    return vec_fp32_relation(fn, eg, "==")

def vec_fp32_close(fn, eg=identity):
    return vec_fp32_condition(fn, f"std::abs(b - {eg('a')}) < 4.0*std::numeric_limits<float>::epsilon()")

def vec_fp64_condition(function_name, condition):
    return f"""
    bool {function_name}(const std::vector<double>& avec, const std::vector<double>& bvec)
    {{
        return std::equal(avec.begin(), avec.end(),
                          bvec.begin(),
                          [](double a, double b)
                          {{
                              return {condition};
                          }});
    }}
    """

def vec_fp64_relation(function_name, expression_generator, relation):
    return vec_fp64_condition(function_name, f"b {relation} {expression_generator('a')}")

def vec_fp64_eq(fn, eg):
    return vec_fp64_relation(fn, eg, "==")

def vec_fp64_close(fn, eg=identity):
    return vec_fp64_condition(fn, f"std::abs(b - {eg('a')}) < 4.0*std::numeric_limits<double>::epsilon()")
