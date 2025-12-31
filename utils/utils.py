def list_to_csv(data : list, file : str = "output.csv", delimiter : str = ",") -> None:
    """
    Write a line of data to a CSV file.

    :param data: list
    :param file: str
    :param delimiter: str

    :return: None
    """
    data = delimiter.join(data) + "\n"

    with open(file, "a") as f:
        f.write(data)