""" A simple date conversion to days """
from datetime import datetime as dt
import preprocess as pre


def get_yday(date):
    ''' Returns the day of the year from the given date '''
    return date.timetuple().tm_yday

fakes, reals = pre.read_data()
print(fakes.date)
print(reals.date)

fmt_dates = [dt.strptime(date, '%B %d, %Y') for date in fakes.date]
fmt_dates.sort()
yday = map(get_yday, fmt_dates)
for day in yday:
    print(day)


deltas = [0]
for curr, nex in zip(fmt_dates[:-1], fmt_dates[1:]):
    delta = nex - curr
    deltas.append(delta.days)

for delta in deltas:
    print(delta)
