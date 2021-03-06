-- Using table: dvdrentals.public.rental

-- We're cracking down on overdue rentals.
-- We have little power over this so we're going to 
-- take a passive aggressive route.  For anyone,
-- who has had their rental for over 1 week, label them with
-- "i'm lame and i turn things in late". For those who've had
-- their dvd for under 24 hours, label them with "newbie".
-- Make up a label or leave the remaining unlabeled.
SELECT customer_id
	, rental_date
	, return_date - rental_date as time_rented
	, CASE 
		WHEN return_date - rental_date > '24 hours' THEN 'late'
		ELSE 'good'
	END as late_status
FROM rental;


-- Everyone knows Garfield hates Mondays.  Garfield also happens
-- to manage this DVD rental place.  Everyone that rents DVDs on
-- Mondays is dead to him.  Return all records where the rental 
-- date is not a Monday.  Include a column showing the day of the
-- week in this output.
-- reference: https://www.postgresql.org/docs/9.6/functions-formatting.html

SELECT rental_date
	, return_date
	, TO_CHAR(rental_date, 'MM/DD/YYYY') AS formatted_rental_date
	, CONCAT(
		TRIM(TO_CHAR(return_date, 'Month'))
		, TO_CHAR(return_date, ' DD, YYYY')) 
	AS formatted_return_date
FROM rental;





-- All of our rentals are due a week after their rental date.
-- Create a column showing the due date for each rental.
-- Create a column indicating whether or not the dvd is overdue.
-- (i know thats kind of like the first prompt, 
--  but its hard to make up unique interesting prompts)



-- Almost all programmers will agree that the best 
-- way to write a date is YYYY-MM-DD.  However, your
-- average US citizen might not agree. Let's malicously
-- comply with society by showing the rental date as
-- month_number/day_number/year and the return date as 
-- month_name day_number, year.
-- reference: https://www.postgresql.org/docs/9.6/functions-formatting.html



-- Make up a prompt to use different techniques we've
-- used so far and bring it to workshop to challenge
-- everyone.  It can use whatever database & table you'd like.


