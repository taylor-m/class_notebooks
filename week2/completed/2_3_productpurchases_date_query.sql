SELECT created_at
	, created_at + INTERVAL '30 Days' AS purchase_plus_thirty
	, created_at - INTERVAL '90 Days' AS purchase_minus_ninety
	, TO_CHAR(created_at, 'DD Mon YYYY') AS formatted_date
	, TO_CHAR(created_at, 'HH:MM') purchase_to_hour
	, TO_CHAR(created_at, 'YYYY') AS purchase_year
	, TO_CHAR(created_at, 'Month') AS purchase_month
	, TO_CHAR(created_at, 'Day') AS purchase_day
	, TO_CHAR(created_at, 'D') AS purchase_weekday
	, TO_CHAR(created_at, 'W') AS purchase_week
from purchases;
	
	
	
	