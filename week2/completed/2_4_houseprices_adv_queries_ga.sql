-- Create a permanent table called normal that contains all property sales where the sale condition was Normal.
CREATE TABLE IF NOT EXISTS normal AS
SELECT *
FROM houseprices
WHERE salecondition = 'Normal';

-- Drop the normal table you just created and then recreate it again using the name normal_sales.
DROP TABLE IF EXISTS normal;

CREATE TABLE IF NOT EXISTS normal_sales AS
SELECT *
FROM houseprices
WHERE salecondition = 'Normal';


-- From the newly-created normal_sales table, delete all the records where the neighborhood is Veenker.
DELETE FROM normal_sales
WHERE neighborhood = 'Veenker';


-- Re-insert the Veenker neighborhood normal sale condition records.
INSERT INTO normal_sales
SELECT *
FROM houseprices
WHERE neighborhood = 'Veenker'
	AND salecondition = 'Normal';

-- For all properties where the sale price was greater than $300,000, update the 
-- overall condition to be equal to the overall quality.
UPDATE normal_sales
SET overallcond = overallqual
WHERE saleprice > 300000;

-- Using your choice of subquery, temp table, or CTE, calculate average sale prices by 
-- neighborhood and then compute what percentage more or less every property sold for 
-- compared to the average for their neighborhood.

WITH n_avg AS (
	SELECT neighborhood
		, ROUND(AVG(saleprice)) AS avg_saleprice
	FROM houseprices
	GROUP BY neighborhood
)
SELECT hp.id AS house_id
	, hp.saleprice AS house_saleprice
	, n_avg.neighborhood AS house_neighborhood
	, n_avg.avg_saleprice AS neighborhood_avg
	, ROUND(((saleprice-n_avg.avg_saleprice)/n_avg.avg_saleprice * 100), 2) AS perc_diff
FROM houseprices AS hp
INNER JOIN n_avg ON hp.neighborhood = n_avg.neighborhood;
















