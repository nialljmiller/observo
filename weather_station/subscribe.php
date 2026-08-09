<?php
// Enable error reporting for debugging
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Initialize message variable
$message = '';
$success = false;

// Process form submission
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['email'])) {
    $email = filter_var($_POST['email'], FILTER_SANITIZE_EMAIL);
    
    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $message = "Invalid email format";
    } else {
        // Store in the specified directory
        $subscribers_file = '/media/bigdata/subscribers.txt';
        
        // Ensure directory exists
        if (!file_exists(dirname($subscribers_file))) {
            mkdir(dirname($subscribers_file), 0755, true);
        }
        
        // Read existing emails
        $current_emails = file_exists($subscribers_file) ? 
            file($subscribers_file, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) : [];
        
        if (!in_array($email, $current_emails)) {
            // Try to write to the file
            if (file_put_contents($subscribers_file, $email . PHP_EOL, FILE_APPEND)) {
                $message = "Subscribed successfully! You'll receive network alerts.";
                $success = true;
            } else {
                $message = "Error: Could not save your subscription. Please try again later.";
            }
        } else {
            $message = "You're already subscribed!";
            $success = true;
        }
    }
}
?>
<!DOCTYPE html>
<html>
<head>
    <title>Subscribe to Network Alerts</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }
        .form-container {
            background: #f8f8f8;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .success { 
            color: green;
            padding: 10px;
            background-color: #e7f7e7;
            border-radius: 4px;
        }
        .error { 
            color: #d32f2f;
            padding: 10px;
            background-color: #ffebee;
            border-radius: 4px;
        }
        input[type="email"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            background: #45a049;
        }
        h2 {
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <h2>Subscribe to Network Alerts</h2>
        
        <?php if (!empty($message)): ?>
            <p class="<?php echo $success ? 'success' : 'error'; ?>">
                <?php echo htmlspecialchars($message); ?>
            </p>
        <?php endif; ?>
        
        <form method="post">
            <label for="email">Enter your email address:</label>
            <input type="email" id="email" name="email" required 
                   placeholder="youremail@example.com">
            <button type="submit">Subscribe</button>
        </form>
    </div>
</body>
</html>
