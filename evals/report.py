# Copyright (c) Alibaba, Inc. and its affiliates.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class GenerateReport:

    def __init__(self, df=None):
        self._df = df

    def gen_mock_report(self):
        data = pd.DataFrame(np.random.randint(1, 5, size=(5, 3)), columns=list('ABC'))
        plt.figure()
        data.plot(kind='bar')
        plt.show()

    def gen_report(self):
        ...


if __name__ == '__main__':
    report = GenerateReport()
    report.gen_mock_report()
