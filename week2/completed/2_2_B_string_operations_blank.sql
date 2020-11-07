-- To answer these use the table:
-- baseball.public.people

-- Request from client:
--    We're going to be interviewing players and we want
--    some text to put on screen so viewers know who they are.
--    Can you create 2 text fields for this? (1 for each line we'll show on screen)
--     * The first text field should have the players name and their age like:
--          "David Aardsma (age: 31)"
--     * The second line should have the handedness:
--          "Bats: R; Throws: R"
--  Please name these fields to indicate which is line 1 and line 2.
--  We're only going to use this overlay for 2020, so there's no need to future proof.
SELECT CONCAT(namefirst, ' ', namelast, ' (age: ', 2020 - birthyear, ')') as line_1,
	CONCAT('Bats: ', bats, '; Throws: ', throws) as line_2
FROM people;


-- Request from client:
--  I can't explain it, but our new CEO says we need to show the players
--  names starting with the letter B.  Can you write a query to output the 
--  players' first and last names with the first letter of each replaced with a B?
--  Name these outputs bamefirst and bamelast.
SELECT CONCAT('B', SUBSTRING(namefirst, 2)) as bamefirst
	,CONCAT('B', SUBSTRING(namelast, 2)) as bamelast
FROM people;

-- Request from client:
--  Ok, that last CEO didn't last long.... 
--  after the B fiasco they were replaced.
--  The new CEO wants to go the other direction.  
--  Can you remove every B from the player names (case insensitive, all bs MUST go)
--  Might be an overcorrection but ¯\_(ツ)_/¯...
--  To ensure that this is working, will you output both the original
--  names and the 'cleaned' names.  Additionally, filter the output
--  to only show names that orginally had Bs in them.

-- 1. [x] find names with Bs or bs in them & filter
-- 2. [] create fields of their names with Bs removed
-- 3. [] output namefirst & namelast
SELECT namefirst
	, namelast
	, REPLACE(REPLACE(namefirst, 'B', ''), 'b', '') as no_b_first
	, REPLACE(REPLACE(namelast, 'B', ''), 'b', '') as no_b_last
FROM people
WHERE namefirst ILIKE '%B%'
	OR namelast ILIKE '%B%';

-- ok... no more made up context to give. so this ones real.
-- i think right handed people are better
-- to reflect this, put the first/last names of players
-- who bat and throw right handed in all caps. anyone who uses
-- their left hand for anything should have their names written in all lower case.
-- please provide a query to assert right hand dominance
-- (return the bats & throws so i can double check the work)
-- for an extra challenge, right this query with only your right hand
SELECT bats
	, throws
	, CASE
		WHEN bats = 'R' AND throws = 'R' THEN UPPER(namefirst || ' ' || namelast)
		ELSE LOWER(namefirst || ' ' || namelast)
	END
FROM people;

