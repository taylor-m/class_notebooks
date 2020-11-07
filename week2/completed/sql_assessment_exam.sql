-- 1. Write a query that allows you to inspect the schema of the naep table. Return only the column_name and data_type fields.

SELECT column_name
	, data_type
FROM information_schema.columns
WHERE table_name = 'naep';


-- 2. Using the style conventions recommended in this course, write a query to select the first fifty records of the naep table. 
-- The answer will only be marked correct if you follow the style conventions recommended in this course.
-- Continue to follow these conventions in this exam and beyond.
SELECT *
FROM naep
LIMIT 50;

-- 3. Write a query that returns summary statistics for avg_math_4_score by state. Make sure to sort the results alphabetically by state name.
SELECT state 
	, COUNT(avg_math_4_score)
	, AVG(avg_math_4_score)
	, MIN(avg_math_4_score)
	, MAX(avg_math_4_score)
FROM naep
GROUP BY state
ORDER BY state;

-- 4. Write a query that alters the previous query so that it returns only the summary statistics for avg_math_4_score by 
-- state with differences in max and min values that are greater than 30. This time, you should use aliases to rename 
-- the max and min calculations to maximum and minimum.


SELECT state 
	, COUNT(avg_math_4_score)
	, AVG(avg_math_4_score)
	, MIN(avg_math_4_score) AS minimum
	, MAX(avg_math_4_score) AS maximum
FROM naep
GROUP BY state
HAVING (MAX(avg_math_4_score) - MIN(avg_math_4_score)) > 30;

-- 5. Write a query that returns a field called bottom_10_states that lists the states in the bottom 10 for avg_math_4_score 
-- in the year 2000.
SELECT state AS bottom_10_states
FROM naep
WHERE year = 2000
ORDER BY avg_math_4_score
LIMIT 10;


-- 6. Write a query that calculates the average avg_math_4_score rounded to the nearest 2 decimal places over all states in the 
-- year 2000. The resulting variable should still be named avg_math_4_score.
SELECT ROUND(AVG(avg_math_4_score), 2) as avg_math_4_score
FROM naep
WHERE year = 2000;

-- 7. Write a query that returns a field called below_average_states_y2000 that lists all states with an avg_math_4_score less 
-- than the average over all states in the year 2000.
SELECT state AS below_average_states_y2000
FROM naep
WHERE avg_math_4_score < (SELECT AVG(avg_math_4_score) FROM naep WHERE year = 2000)
	AND year = 2000;
	
-- 8. Write a query that returns a field called scores_missing_y2000 that lists any states with missing values in the 
-- avg_math_4_score column of the naep data table for the year 2000.
SELECT state AS scores_missing_y2000
FROM naep
WHERE avg_math_4_score IS null
	AND year = 2000;


-- 9. For this challenge, you're being asked to display the average math score alongside the total expenditure to see if there 
-- is any relationship between the two. We are not focusing on doing correlations in this challenge, but rather, this challenge 
-- will test your skills with using JOIN.
-- To accomplish your task, write a query that returns, for the year 2000, the state,avg_math_4_score, and total_expenditure fields 
-- from the naep table left outer joined on the finance table, using id as the key and ordered by total_expenditure, from largest 
-- to smallest.
-- Be sure to round avg_math_4_score to the nearest 2 decimal places, keeping the variable name as is using an alias, and then 
-- filter out NULL from avg_math_4_scores in order to see the results more clearly.
SELECT naep.state
	, ROUND(naep.avg_math_4_score, 2) AS avg_math_4_score
	, finance.total_expenditure
FROM naep
LEFT JOIN finance ON finance.id = naep.id
WHERE naep.year = 2000
	AND naep.avg_math_4_score IS NOT NULL
ORDER BY finance.total_expenditure DESC;


