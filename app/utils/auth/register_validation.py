import re

# Make a regular expression
# for validating an Email
EMAIL_REGEX: re.Pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{1,7}\b")


# Define a function for
# for validating an Email
def is_email_valid_format(email: str) -> bool:
    return True if EMAIL_REGEX.fullmatch(email) is not None else False


def is_email_length_in_range(email: str) -> bool:
    return True if 6 <= len(email) <= 50 else False


def is_password_length_in_range(password: str) -> bool:
    return True if 6 <= len(password) <= 100 else False


# Driver Code
if __name__ == "__main__":
    for email in ("ankitrai326@gmail.com", "my.ownsite@our-earth.org", "ankitrai326.com", "aa@a.a"):
        # calling run function
        print(is_email_valid_format(email))
