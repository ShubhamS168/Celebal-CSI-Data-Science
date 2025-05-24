def lower_triangle(n):
    try:
        n = int(n)
        if n <= 0:
            raise ValueError("Input must be a positive integer.")
        for i in range(n, 0, -1):
            print("* " * i)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    rows = input("Enter number of rows for lower triangle: ")
    lower_triangle(rows)
