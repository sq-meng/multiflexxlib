import multiflexxlib as mfl
import pytest


def test1():
    filename_list = ['test/067777', 'test/067778', '067780']
    a3_offset_value = 5.5
    a3_offset_list = [3.3, 2.2, 1.1]
    a3_offset_list_wrong = [3.2, 1.1]
    a3_offset_dict = {'067777': 3.2, '067780': 1.1}
    a3_offset_dict_num = {67777: 3.2, '067780': 1.1}
    assert mfl.mftools._expand_offset_parameter(a3_offset_list, filename_list) == [3.3, 2.2, 1.1]
    assert mfl.mftools._expand_offset_parameter(a3_offset_value, filename_list) == [5.5, 5.5, 5.5]
    assert mfl.mftools._expand_offset_parameter(a3_offset_dict, filename_list) == [3.2, 0.0, 1.1]
    assert mfl.mftools._expand_offset_parameter(a3_offset_dict_num, filename_list) == [3.2, 0.0, 1.1]
    with pytest.raises(ValueError):
        mfl.mftools._expand_offset_parameter(a3_offset_list_wrong, filename_list)
    with pytest.raises(TypeError):
        mfl.mftools._expand_offset_parameter('abc', filename_list)

