def greedy_coin_change(amount, coins=[25, 10, 5, 1]):
    """
    Greedy heuristic for making change
    Always picks the largest coin that fits
    
    amount: Total amount to make change for
    coins: Available coin denominations (sorted largest to smallest)
    
    Returns: Dictionary showing count of each coin used
    
    Example: 
    greedy_coin_change(41) 
    → {25: 1, 10: 1, 5: 1, 1: 1} (1 quarter, 1 dime, 1 nickel, 1 penny)
    """
    result = {}
    
    for coin in coins:
        if amount >= coin:
            count = amount // coin  # How many of this coin can we use?
            result[coin] = count
            amount -= coin * count  # Subtract the value we've used
    
    return result

# Test
if __name__ == "__main__":
    print("=== Testing Greedy Coin Change ===")
    
    test_amounts = [41, 63, 99, 100]
    
    for amt in test_amounts:
        coins_used = greedy_coin_change(amt)
        total_coins = sum(coins_used.values())
        print(f"\nAmount: {amt} cents")
        print(f"Coins used: {coins_used}")
        print(f"Total number of coins: {total_coins}")