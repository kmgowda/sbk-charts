import re
from collections import OrderedDict

import src.sheets.constants as sheets_constants
from src.charts import constants


def is_r_num_sheet(name):
    """Return True if the worksheet name matches the R-prefix numeric pattern.

    Parameters
    - name (str): worksheet name

    Returns
    - bool: whether name matches the pattern 'R<digits>'
    """
    return re.match(r'^' + sheets_constants.R_PREFIX + r'\d+$', name)


def is_t_num_sheet(name):
    """Return True if the worksheet name matches the T-prefix numeric pattern.

    Parameters
    - name (str): worksheet name

    Returns
    - bool: whether name matches the pattern 'T<digits>'
    """
    return re.match(r'^' + sheets_constants.T_PREFIX + r'\d+$', name)


def get_columns_from_worksheet(ws):
    """Return an OrderedDict mapping column header names to Excel column
    numbers from the first row of the provided worksheet.

    Parameters
    - ws: openpyxl worksheet

    Returns
    - OrderedDict[str, int]: mapping of header->column index
    """
    ret = OrderedDict()
    for cell in ws[1]:
        if cell.value:
            ret[cell.value] = cell.column
    return ret


def get_time_unit_from_worksheet(ws):
    """Read and return the latency time unit from the supplied worksheet.

    Parameters
    - ws: openpyxl worksheet (expected to be an R sheet)

    Returns
    - str: time unit in uppercase (e.g. 'MS', 'US')
    """
    names = get_columns_from_worksheet(ws)
    return str(ws.cell(row=2, column=names[constants.LATENCY_TIME_UNIT]).value).upper()


def get_storage_name_from_worksheet(ws):
    """Return the storage name value from the second row of the worksheet.

    Parameters
    - ws: openpyxl worksheet

    Returns
    - str: storage name uppercased
    """
    names = get_columns_from_worksheet(ws)
    return str(ws.cell(row=2, column=names[constants.STORAGE]).value).upper()


def get_action_name_from_worksheet(ws):
    """Return the action name value from the second row of the worksheet.

    Parameters
    - ws: openpyxl worksheet

    Returns
    - str: action name
    """
    names = get_columns_from_worksheet(ws)
    return str(ws.cell(row=2, column=names[constants.ACTION]).value)