-- Write a query that joins the purchase_items and product tables. The query should return 
-- the purchase ID, title, quantity, and price fields for all returned laptop computers.
SELECT purchase_id
	, title
	, quantity
	, pi.price
FROM purchase_items AS pi
JOIN products ON products.id = pi.product_id
WHERE pi.state = 'Returned';

-- Write a query that joins all 4 tables in the database and returns all MP3 Player purchase 
-- line items that are in pending status and were ordered from the state of Florida (FL) by email 
-- address Derek.Knittel@gmail.com. The query should return the purchase ID, the order status, 
-- the customer name, the state, the product name, the order quantity, the order price, and the customer email.
SELECT DISTINCT purchase_id
	, purchase_items.state
	, purchases.name
	, purchases.state
	, products.title
	, purchase_items.quantity
	, purchase_items.price
	, users.email
FROM products
JOIN purchase_items ON products.id = purchase_items.product_id
JOIN purchases ON purchase_items.purchase_id = purchases.id
JOIN users ON purchases.user_id = users.id
WHERE products.title = 'MP3 Player'
	AND purchase_items.state = 'Pending'
	AND purchases.state = 'FL'
	AND users.email = 'Derek.Knittel@gmail.com';
	