#!/usr/bin/env python3

from my_solution import predict

# 测试用例
def test_solution():
    origin_list = "中性"

    assert predict("您好！很高兴为您服务。") == origin_list