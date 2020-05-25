import pandas as pd


def PaO2_to_SpO2(pO2):
    """
    conversion based on
    Severinghaus, J. W. Simple, accurate equations for human blood O2 dissociation computations.
    J Appl Physiol. 46(3): 599-602. 1979.
    also, ref: https://www.intensive.org/epic2/Documents/Estimation%20of%20PO2%20and%20FiO2.pdf
    :param pO2:
    :return:
    """
    return (23400 * (pO2**3 + 150 * pO2)**-1 + 1)**-1 * 100


def read_data(data_file, sep='\t', column_mapping=None, partial_to_saturation_col=None):
    x = pd.read_csv(data_file, sep=sep)
    if PaO2_to_SpO2 is not None:
        x[partial_to_saturation_col] = [PaO2_to_SpO2(v) for v in x[partial_to_saturation_col]]
    if column_mapping is not None:
        x.rename(columns=column_mapping, inplace=True)
    return x


if __name__ == "__main__":
    print(PaO2_to_SpO2(44))