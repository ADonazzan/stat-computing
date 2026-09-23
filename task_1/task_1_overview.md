# Summary of data analysis in Task 1

## Data
The flights csv contains 187432 observations for 19 columns, with some missing data. For instance, about 3,450 flights have no departure time and about 4,000 have no arrival time. To deal with this, I add a `status` column, which is a factor indicating the reason for missingness: `0 = normal, 1 = cancelled (no dep_time), 2 = diverted (departed but no arr_time), 3 = some other missing value`.

I grouped by origin and destination airport in order to obtain route summaries. For each route, I computed:
- The number of flights
- The median delay among late departures
- The share of flights departing late and the share arriving late (15+ minutes, as per FAA threshold)
- The share of cancelled or diverted

I then filtered flights between Philadelphia and Chicago [PHL and ORD/MDW], that were delayed more than 2 hours on departure or arrival.

## New measures, aggregations and joins

I derive the variables as requested, and compute ground speed as distance/air time, as the column in planes is almost entirely empty. The aggregations following this are used to compute mean departure delay, total flights, and other similar measures. Finally, i join multiple dataframes to obtain airport names and time zones for origin and destination airports.

## Question about the data
I want to understand what the best time of the year to fly west from Pittsburgh is, once weather at departure is taken into account. To do so, I only considered departures to airports west of longitude -100°, and matched each flight to PIT's weather at its scheduled departure hour. I considered the weather to be severe if it's in the worse 5% of rainy hours, wind gusts, or visibility. A flight counts as a bad outcome if it is more than 15 minutes late, or it was cancelled or diverted.

The best months to fly west, in the sense of the least delays or cancellations, are October and November, where 12% of total flights had a bad outcome and 9% ran into severe weather. In June, February and March, 22% to 25% of flights recorded a bad outcome, with 20% of the total experiencing severe weather.