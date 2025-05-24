def upper_triangle(n):
    try:
        n = int(n)
        if n <= 0:
            raise ValueError("Input must be a positive integer.")
        for i in range(1, n + 1):
            print("* " * i)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    rows = input("Enter number of rows for upper triangle: ")
    upper_triangle(rows)
