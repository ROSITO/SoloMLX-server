from datetime import date, timedelta


def last_n_days(n: int):
    end_date = date.today()
    start_date = end_date - timedelta(days=n)
    return start_date, end_date
