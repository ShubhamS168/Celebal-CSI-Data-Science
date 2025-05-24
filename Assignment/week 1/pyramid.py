def pyramid(n):
    try:
        n = int(n)
        if n <= 0:
            raise ValueError("Input must be a positive integer.")
        for i in range(n):
            spaces = "  " * (n - i - 1)
            stars = "* " * (2 * i + 1)
            print(spaces + stars)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    rows = input("Enter number of rows for pyramid: ")
    pyramid(rows)
