# Add New Chart

## Overview
This skill guides through adding a new chart type to sbk-charts.

## When to use this skill
Use this skill when:
- Adding a new visualization for benchmark data
- Creating a comparison chart for existing metrics
- Adding a new type of performance analysis chart

## Prerequisites
- Read `docs/AGENT_RECIPES.md` for the "Add a new chart" recipe
- Understand the existing chart patterns in `src/charts/charts.py` and `src/charts/multicharts.py`
- Have a clear understanding of the data structure in R/T sheets

## Steps

### 1. Add column constants (if new metrics)
Update `src/charts/constants.py`:
```python
NEW_METRIC = "New Metric Name"  # Must match CSV header exactly
```

### 2. Determine chart type
Decide if the chart is:
- **Per-run chart** (in `src/charts/charts.py`): One chart per R-sheet
- **Multi-run chart** (in `src/charts/multicharts.py`): Combined chart across all R-sheets

### 3. Implement per-run chart (if applicable)
Add method to `SbkCharts` class in `src/charts/charts.py`:
```python
def create_<chart_name>_graph(self):
    """Create <chart description>.

    Returns
    - Worksheet or None
    """
    chart = self.create_<chart_type>_chart(title, x_label, y_label, width, height)

    # Add data series
    for name in self.wb.sheetnames:
        if is_r_num_sheet(name):
            ws = self.wb[name]
            # ... add series logic

    if chart is not None:
        sheet = self.wb.create_sheet("Chart_Name")
        sheet.add_chart(chart)
        return sheet
    return None
```

### 4. Implement multi-run chart (if applicable)
Add method to `SbkMultiCharts` class in `src/charts/multicharts.py`:
```python
def create_multi_<chart_name>_graph(self):
    """Create combined <chart description> across R-sheets.

    Returns
    - Worksheet or None
    """
    chart = self.create_<chart_type>_chart(title, x_label, y_label, width, height)

    for name in self.wb.sheetnames:
        if is_r_num_sheet(name):
            ws = self.wb[name]
            # ... add series logic

    if chart is not None:
        sheet = self.wb.create_sheet("Total_<Chart_Name>")
        sheet.add_chart(chart)
        return sheet
    return None
```

### 5. Wire into the graph generation pipeline
Add the call to the appropriate location:
- For per-run: in `SbkCharts.create_graphs()` in `src/charts/charts.py`
- For multi-run: in `SbkMultiCharts.create_graphs()` in `src/charts/multicharts.py`

### 6. Test the chart
```bash
sbk-charts -i samples/charts/sbk-file-read.csv -o /tmp/test-chart.xlsx
```

Open the output Excel and verify the chart appears and looks correct.

## Chart Types Available

### Line Chart
```python
chart = self.create_line_chart(title, x_label, y_label, width, height)
```

### Bar Chart
```python
chart = self.create_bar_chart(title, x_label, y_label, width, height)
```

### Scatter Chart
```python
chart = self.create_scatter_chart(title, x_label, y_label, width, height)
```

## Data Series Patterns

### Single metric per sheet
```python
series = Reference(ws, min_col=col, min_row=2, max_col=col, max_row=ws.max_row)
chart.append(Series(series, title=f"{prefix} - Metric"))
```

### Multiple metrics per sheet
```python
for metric in metrics:
    col = cols[metric]
    series = Reference(ws, min_col=col, min_row=2, max_col=col, max_row=ws.max_row)
    chart.append(Series(series, title=f"{prefix} - {metric}"))
```

### Combined across sheets
```python
for name in self.wb.sheetnames:
    if is_r_num_sheet(name):
        ws = self.wb[name]
        prefix = name + "_" + get_storage_name_from_worksheet(ws)
        # ... add series
```

## Naming Conventions
- Method: `create_<chart_type>_graph` (per-run) or `create_multi_<chart_type>_graph` (multi-run)
- Sheet name: `<ChartName>` (per-run) or `Total_<ChartName>` (multi-run)
- Use underscores in method names, PascalCase in sheet names

## Verification
- Chart appears in the output Excel file
- Chart has correct title and labels
- Data series are properly labeled
- Chart renders correctly in Excel
- No errors in the console output
