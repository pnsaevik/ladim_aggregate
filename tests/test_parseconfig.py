from ladim_aggregate import parseconfig


class Test_parse_config:
    def test_output_equals_input_when_unknown_keywords(self):
        conf = dict(unknown_keyword="some_values")
        conf_out = parseconfig.parse_config(conf)
        assert conf_out["unknown_keyword"] == "some_values"

    def test_dont_change_bin_if_present_and_in_filesplit_dims(self):
        conf = dict(
            bins=dict(x=[1, 2, 3], farm_id=[12345, 23456]),
            filesplit_dims=['farm_id'],
        )
        conf_out = parseconfig.parse_config(conf)
        assert conf_out['bins']['farm_id'] == [12345, 23456]

    def test_adds_groupby_bin_on_top_if_missing_and_in_filesplit_dims(self):
        conf = dict(
            bins=dict(x=[1, 2, 3]),
            filesplit_dims=['farm_id'],
        )
        conf_out = parseconfig.parse_config(conf)
        assert conf_out['bins']['farm_id'] == 'group_by'
        assert list(conf_out['bins'].keys()) == ['farm_id', 'x']
