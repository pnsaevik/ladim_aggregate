from ladim_aggregate import parseconfig


class Test_parse_config:
    def test_output_equals_input_when_unknown_keywords(self):
        conf = dict(this_is_an_unknown_keyword="some_values")
        conf_out = parseconfig.parse_config(conf)
        assert conf_out == conf
