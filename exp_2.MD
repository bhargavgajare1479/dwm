# **Practical 2: OLAP Operations using SQL**

### **1. Roll Up**

```sql
SELECT 
  c.Country, 
  SUM(o.Amount) AS Total_Sales
FROM Orders o
JOIN Customers c 
  ON o.Customer_id = c.Customer_id
GROUP BY c.Country

UNION ALL

SELECT 
  'Grand Total' AS Country, 
  SUM(o.Amount)
FROM Orders o;
```

---

### **2. Drill Down**

```sql
SELECT 
  c.Country, 
  c.first_name, 
  SUM(o.amount) AS Total_Sales
FROM Orders o
JOIN Customers c 
  ON o.Customer_id = c.Customer_id
GROUP BY 
  c.country, 
  c.first_name
ORDER BY 
  c.country;
```

---

### **3. Slice**

```sql
SELECT 
  c.first_name,
  c.country,
  o.item,
  o.amount,
  s.status
FROM Orders o
JOIN Customers c 
  ON o.customer_id = c.customer_id
JOIN Shippings s 
  ON s.customer = c.customer_id
WHERE s.status = 'Delivered';
```

---

### **4. Dice**

```sql
SELECT 
  c.first_name, 
  c.country, 
  o.item, 
  o.amount, 
  s.status
FROM Orders o
JOIN Customers c 
  ON o.customer_id = c.customer_id
JOIN Shippings s 
  ON s.customer = c.customer_id
WHERE 
  c.country = 'USA' 
  AND o.amount < 300 
  AND s.status = 'Pending';
```

---

### **5. Pivot**

```sql
SELECT 
  c.country,
  s.status,
  SUM(o.amount) AS total_amount
FROM Orders o
JOIN Customers c 
  ON o.customer_id = c.customer_id
JOIN Shippings s 
  ON s.customer = c.customer_id
GROUP BY 
  c.country, 
  s.status;
```

```sql
SELECT 
  c.country,
  SUM(CASE WHEN s.status = 'Delivered' THEN o.amount ELSE 0 END) AS Delivered_Total,
  SUM(CASE WHEN s.status = 'Pending' THEN o.amount ELSE 0 END) AS Pending_Total,
  SUM(CASE WHEN s.status = 'Cancelled' THEN o.amount ELSE 0 END) AS Cancelled_Total
FROM Orders o
JOIN Customers c 
  ON o.customer_id = c.customer_id
JOIN Shippings s 
  ON s.customer = c.customer_id
GROUP BY c.country;
```