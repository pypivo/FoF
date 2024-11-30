from openpyxl import load_workbook


def read_coefficients_row_from_xlsx(file_path):
    """
    Считывает данные из указанного xlsx-файла в словарь.
    Ключи словаря — названия столбцов, значения — числа из строки.

    :param file_path: Путь к xlsx файлу.
    :param row_number: Номер строки для чтения (начиная с 1).
    :return: Словарь, где ключи — названия столбцов, значения — числа из строки.
    """
    try:
        row_number = int(input("Введите номер строки из xlsx файла с коэффициентами: "))
        workbook = load_workbook(file_path)
        sheet = workbook.active

        headers = [cell.value for cell in sheet[1]]
        row = [cell.value for cell in sheet[row_number]]

        result = {header: float(value) for header, value in zip(headers, row) if isinstance(value, (int, float))}
        return result
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
    except IndexError:
        print(f"Строка с номером {row_number} отсутствует в файле.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

    return None
