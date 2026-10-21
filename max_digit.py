n = int(input())

max_digit = -1
count = 0

while n > 0:
    d = n % 10
    if d > max_digit:
        max_digit = d
        count = 1
    elif d == max_digit:
        count += 1
    n //= 10

print(max_digit)
print(count)
