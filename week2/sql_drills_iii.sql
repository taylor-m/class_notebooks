-- 1. Write a query that returns the namefirst and namelast fields of the people table, along with the inducted 
-- field from the hof_inducted table. All rows from the people table should be returned, and NULL values for 
-- the fields from hof_inducted should be returned when there is no match found.
SELECT namefirst
	, namelast
	, hof_inducted.inducted
FROM people
LEFT JOIN hof_inducted ON people.playerid = hof_inducted.playerid;


-- 2. Write a query that returns the yearid, playerid, teamid, and salary fields from the salaries table, 
-- along with the category field from the hof_inducted table.
-- Keep only the records that are in both salaries and hof_inducted.
-- Hint: While a field named yearid is found in both tables, don’t JOIN by it. You must, however, 
-- explicitly name which field to include.
SELECT salaries.yearid
	, salaries.playerid
	, salaries.teamid
	, salaries.salary
	, hof_inducted.category
FROM salaries
INNER JOIN hof_inducted ON hof_inducted.playerid = salaries.playerid;



-- 3. Write a query that returns the playerid, yearid, teamid, lgid, and salary fields from the salaries table 
-- and the inducted field from the hof_inducted table.
-- Keep all records from both tables.
SELECT salaries.playerid
	, salaries.yearid
	, salaries.teamid
	, salaries.lgid
	, salaries.salary
	, hof_inducted.inducted
FROM salaries
FULL JOIN hof_inducted ON hof_inducted.playerid = salaries.playerid;


-- 4. There are 2 tables, hof_inducted and hof_not_inducted, indicating successful and unsuccessful inductions into 
-- the Baseball Hall of Fame, respectively.
-- Combine these 2 tables and return a single field that shows a list of playerids from both input tables. Keep only unique records.
SELECT playerid
FROM hof_inducted
UNION
SELECT playerid
FROM hof_not_inducted;


-- 5. Write a query that returns the last name, first name (see people table), and total recorded salaries for all 
-- players found in the salaries table.
SELECT people.namelast
	, people.namefirst
	, SUM(salaries.salary) AS total_salary
FROM salaries
INNER JOIN people ON people.playerid = salaries.playerid
GROUP BY people.playerid, people.namelast, people.namefirst;


-- 6. Write a query that returns all records from the hof_inducted and hof_not_inducted tables that include playerid, 
-- yearid, namefirst, and namelast.
-- Hint: Each FROM statement will include a LEFT OUTER JOIN!
SELECT hof_inducted.playerid
	, hof_inducted.yearid
	, namefirst
	, namelast
FROM hof_inducted
LEFT OUTER JOIN people ON people.playerid = hof_inducted.playerid
UNION ALL
SELECT hof_not_inducted.playerid
	, hof_not_inducted.yearid
	, namefirst
	, namelast
FROM hof_not_inducted
LEFT OUTER JOIN people ON people.playerid = hof_not_inducted.playerid;




-- 7. Return a table including all records from both hof_inducted and hof_not_inducted, and include a new field, namefull, 
-- which is formatted as namelast , namefirst (in other words, the last name, followed by a comma, then a space, then 
-- the first name). 
-- Use the CONCAT() function for this operation instead of the concat operator (||), as they behave 
-- differently when working with NULL values.
-- The query should also return the yearid and inducted fields. Include only records since 1980 from both tables. Sort 
-- the resulting table by yearid, then inducted so that Y comes before N.
-- Finally, sort by the namefull field, A to Z.
SELECT CONCAT(namelast, ', ', namefirst) AS namefull
	, hof_inducted.yearid
	, hof_inducted.inducted
FROM hof_inducted
LEFT JOIN people ON people.playerid = hof_inducted.playerid
WHERE hof_inducted.yearid >= 1980
UNION ALL
SELECT CONCAT(namelast, ', ', namefirst) AS namefull
	, hof_not_inducted.yearid
	, hof_not_inducted.inducted
FROM hof_not_inducted
LEFT JOIN people ON people.playerid = hof_not_inducted.playerid
WHERE yearid >= 1980
ORDER BY yearid, inducted DESC, namefull ASC;


-- 8. Write a query that returns each year's highest annual salary for each teamid, ranked from high to low, along with 
-- the corresponding playerid.
-- Bonus! Return namelast and namefirst in the resulting table. (You can find these in the people table.)


WITH fxn AS (
SELECT yearid
	, teamid
	, MAX(salary) AS max_salary
FROM salaries
GROUP BY teamid, yearid
)

SELECT fxn.max_salary
	, salaries.playerid
	, salaries.yearid
	, salaries.teamid
FROM salaries
LEFT JOIN fxn ON salaries.salary = fxn.max_salary
	AND salaries.yearid = fxn.yearid
	AND salaries.teamid = fxn.teamid
WHERE fxn.max_salary IS NOT NULL
ORDER BY salary DESC;




-- Select birthyear, deathyear, namefirst, and namelast of all the players born since the birth year of 
-- Babe Ruth (playerid = ruthba01).
-- Sort the results by birth year from low to high.

SELECT birthyear
	, deathyear
	, namefirst
	, namelast
FROM people
WHERE birthyear >= (
	SELECT birthyear
	FROM people
	WHERE playerid = 'ruthba01')
ORDER BY birthyear ASC;


-- Using the people table, write a query that returns namefirst, namelast, and a field called usaborn.
-- The usaborn field should show the following: if the player's birthcountry is the USA, then the record is USA. Otherwise, it's non-USA.
-- Order the results by non-USA records first.
SELECT namefirst
	, namelast
	, CASE
		WHEN birthcountry = 'USA' THEN 'USA'
		ELSE 'non-USA'
	END AS usaborn
FROM people
ORDER BY usaborn ASC;


-- Calculate the average height for players throwing with their right hand versus their left hand.
-- Name these fields right_height and left_height, respectively.
SELECT DISTINCT(
	SELECT AVG(height)
	FROM people
	WHERE throws = 'L') AS left_height
	, (
	SELECT AVG(height)
	FROM people
	WHERE throws = 'R') AS right_height
FROM people
GROUP BY throws;


-- Get the average of each team's maximum player salary since 2010. Hint: WHERE will go outside your CTE.
WITH max_sal_by_team_by_year AS
(
SELECT yearid 
	, teamid
	, MAX(salary) AS max_salary
FROM salaries
GROUP BY yearid, teamid
ORDER BY yearid
)
SELECT yearid
	, AVG(max_salary)
FROM max_sal_by_team_by_year
WHERE yearid > 2010
GROUP BY yearid;





