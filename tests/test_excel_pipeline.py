from services.excel_service import ExcelService
from services.validation_service import ValidationService
from services.missing_value_service import MissingValueService
from services.type_converter_service import TypeConverterService

success, msg, df = ExcelService.load_excel("crime_data.xlsx")

if success:

    success, msg, df = ValidationService.validate_columns(df)

    if success:
        success, msg, df = MissingValueService.handle_missing_values(df)

    if success:
        success, msg, df = TypeConverterService.convert_types(df)

print(msg)
