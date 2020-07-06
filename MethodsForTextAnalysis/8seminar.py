def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def fibonnaci_encoding(number):
    n = 1
    while number >= fibonacci(n):
        n += 1
    n -= 1
    remainder = number - fibonacci(n)
    fibonacci_code = [0 for i in range(n)]
    fibonacci_code[n - 2] = 1
    fibonacci_code[n - 1] = 1
    while remainder != 0:
        # print(remainder)
        number = remainder
        n = 1
        while number >= fibonacci(n):
            n += 1
        n -= 1
        remainder = number - fibonacci(n)
        fibonacci_code[n - 2] = 1
    return fibonacci_code


def fibonacci_decode(encoded_number):
    decoded_number = 0
    for n in range(2, len(encoded_number) + 1):
        decoded_number += fibonacci(n) * encoded_number[n - 2]
    # print(decoded_number)
    return decoded_number


# print(fibonacci(2))
for i in range(1, 15):
    print(i)
    encoded = fibonnaci_encoding(i)
    print(encoded)
    decoded = fibonacci_decode(encoded)
    print(decoded)
