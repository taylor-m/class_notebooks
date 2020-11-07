-- Concatenate the fuel and cyl fields together (with a space in between).
-- Replace any occurrences of the word “Small” in the class field with an empty string.
-- Create a field that parses out whether the vehicle is Automatic or Manual from the trans field (without the additional speed info).
-- Filter for records where the drive train info (ex. 2WD, 4WD, AWD, etc.) is listed at the end of the model field.
SELECT CONCAT(fuel, ' ', cyl) as fuel_cyl
	, REPLACE(class, 'Small', '') as class_no_small
	, CASE
		WHEN trans LIKE 'A%' THEN SUBSTRING(trans, 1, 9)
		ELSE SUBSTRING(trans, 1, 6)
	END as trans_class
	, model
FROM vehicles
WHERE model LIKE '%WD';