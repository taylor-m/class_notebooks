-- Using the band_members database

----------------------------------
-- JOINs -------------------------
----------------------------------

-- Explore the 2 tables in the band_members database.
-- How can we combine the information from these tables?
-- ERD - schema
-- PK - primary key = customer.customerID
-- FK - foreign  key = order.customerID


-- Use an join the 2 tables and only keep records
-- where the key was found in both tables
SELECT *
FROM band_instruments 
INNER JOIN band_members ON band_instruments.name = band_members.name;
-- ignored the cases where the names didn't match up
-- 'innerjoin' is the default type of join
-- name column also returns 2 name columns bc we used *


-- SELECT * brings back our key column twice!
-- Be explicit to avoid confusing field naming.
SELECT band_instruments.name, plays, band
FROM band_instruments
INNER JOIN band_members ON band_instruments.name = band_members.name;


-- Make it more readable with aliasing
SELECT bi.name, plays, band
FROM band_instruments AS bi
INNER JOIN band_members AS bm 
	ON bi.name = bm.name;


-- Note, we don't *need* the AS keyword
-- Some RDBMSes actually won't let you use AS to alias tables
SELECT bi.name, plays, band
FROM band_instruments bi
INNER JOIN band_members bm 
	ON bi.name = bm.name;


-- We need to reference what table our name column is coming from
-- since it's in both tables SQL will complain about ambiguity



-- Conversely we could prefix every column if we wanted
-- This isn't needed since band & plays aren't ambiguous



-- We've used an INNER JOIN. What about other JOINS?
-- INNER is implied when just using JOIN,
SELECT bi.name, bm.name, plays, band
FROM band_instruments bi
FULL OUTER JOIN band_members bm
		ON bi.name = bm.name;


-- A FULL OUTER JOIN grabs every row from both tables
-- You'll sometimes see OUTER dropped from this type of JOIN
SELECT bi.name, bm.name, plays, band
FROM band_instruments bi
FULL JOIN band_members bm
		ON bi.name = bm.name;


-- LEFT OUTER JOIN (think venn diagram)
-- you'll often see people drop the OUTER from this 
-- type of JOIN
SELECT bi.name, plays, band
FROM band_instruments AS bi
LEFT JOIN band_members AS bm
ON bi.name = bm.name;


-- RIGHT JOINs work very similarly to LEFT JOINs
SELECT *
FROM band_instruments AS bi
RIGHT JOIN band_members AS bm
ON bi.name = bm.name;


----------------------------------
-- Set operations ----------------
----------------------------------

-- Joins will combine tables horizontally combine
-- tables, sometimes we have a usecase for combining
-- tables vertically.  For this, we can use UNION.
-- Using UNION will only grab unique records
-- This can be thought of as set addition.
-- What are some potential use cases of UNION?



-- UNION ALL will not filter out duplicates




-- Using INTERSECT will stack tables similarly
-- to UNION.  The difference is that INTERSECT
-- only keeps rows common to both tables.
-- What are some potential use cases of INTERSECT?



-- The last set operator we're going to look at is
-- EXCEPT.  This can be thought of as set subtraction.
-- EXCEPT will give us all rows from the first table
-- that are not present in the second.


