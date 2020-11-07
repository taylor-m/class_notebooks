
-- Adds together fixed_acidity, volatile_acidity, and citric_acid
-- Subtract free_sulfur_dioxide from total_sulfur_dioxide and then divide the result by total_sulfur_dioxide.
-- Multiply residual_sugar by alcohol, raise the result to the 4th power, and then take the log (base 10) of that.
-- Round chlorides to two decimal places, multiply by total_sulfur_dioxide, and then obtain the cube root of that.

SELECT fixed_acidity + volatile_acidity + citric_acid AS sum_acidity
	,(total_sulfur_dioxide - free_sulfur_dioxide)/total_sulfur_dioxide AS sulfur_percentage
	,LOG((residual_sugar * alcohol)^(4.0)) AS sugar_log
	,(ROUND(chlorides, 2) * total_sulfur_dioxide)^(1.0/3.0) AS sulfur_chlorides_cbrt
FROM winequality_red;