import polars as pl

path_flights = "data/pa-flights/flights.csv"
flights = pl.read_csv(path_flights, infer_schema_length=10000, null_values=["NA", ""])

# Check what data contains: 187432 observations for 19 columns
flights_shape = flights.shape
flights_schema = flights.schema 

flights.null_count() 
# around 3450 flights with missing departure time, 4k with missing arrival
# there are cancelled flights, those with no arrival are probably flights that got diverted

# I will keep na values, but flag them as cancelled/diverted:
flights = flights.with_columns(
            pl.when(pl.col("dep_time").is_null()).then(1)          # cancelled
              .when(pl.col("arr_time").is_null()).then(2)          # diverted
              .when(pl.any_horizontal(pl.all().is_null())).then(3) # other
              .otherwise(0).alias("status")
            )

city_summary = flights.group_by(["origin", "dest"]).agg(
                    pl.len().alias("total_flights"),
                    pl.col("dep_delay").filter(pl.col("dep_delay") >= 15).median().alias("median_dep_delay"), # mean delay once the flight is late (15 minutes as per FAA)
                    (pl.col("dep_delay") >= 15).mean().alias("perc_dep_delays"), # fraction of flights that left more than 15 minutes late 
                    (pl.col("dep_delay") >= 15).mean().alias("perc_arr_delays"), # fraction of flights that arrived more than 15 minutes late
                    (pl.col("status").is_in([1,2])).mean().alias("cancelled/diverted flights") # fraction of flights that were cancelled or diverted
                ).sort("total_flights", descending=True)

# Selecting Philadelphia and Chicago
PHL = ["PHL"]
CHI = ["ORD", "MDW"] 

phl_chi = flights.filter(
    (pl.col("origin").is_in(PHL) & pl.col("dest").is_in(CHI))
    | (pl.col("origin").is_in(CHI) & pl.col("dest").is_in(PHL))
    ).filter(
        (pl.col("dep_delay") > 120) | (pl.col("arr_delay") > 120)
    ).sort("dep_delay", descending=True)
    
# --------------------------
# Deriving New Measures
# --------------------------

# red-eye flights: leaving after 9pm and arriving before 7am, crossing midnight
flights = flights.with_columns(
    ((pl.col("sched_dep_time") >= 2100) & (pl.col("sched_arr_time") < 700)
        & (pl.col("sched_arr_time") < pl.col("sched_dep_time"))).alias("red_eye")
)

# delay severity
severity = pl.Enum(["none", "minor", "major"])
flights = flights.with_columns(
    pl.when(pl.col("arr_delay") < 15).then(pl.lit("none"))     # on-time according to FAA
      .when(pl.col("arr_delay") < 60).then(pl.lit("minor"))
      .otherwise(pl.lit("major"))
      .cast(severity)
      .alias("delay_severity")
)

# ground speed
flights = flights.with_columns(
    pl.when(pl.col("air_time") > 0)
      .then(pl.col("distance") / pl.col("air_time") * 60)
      .otherwise(None)
      .alias("ground_speed")
)

# --------------------------
# Grouping and Aggregation
# --------------------------

# average departure delay by carrier
carriers = flights.group_by("carrier").agg(
    pl.col("dep_delay").mean().alias("avg_dep_delay")
).sort("avg_dep_delay", descending=True)

# speed by plane: join with planes data
# note: speed data is zero for all obs except for three
planes = pl.read_csv("data/pa-flights/planes.csv", null_values=["NA", ""]).rename({"year": "plane_year"})

speed_by_plane = flights.join(planes, on="tailnum", how="inner").group_by(["type", "engine", "engines"]
        ).agg(
        pl.len().alias("total_flights"),
        pl.col("ground_speed").mean().alias("avg_ground_speed")
        )

# monthly total flights
monthly_flights = flights.group_by(["origin", "month"]).agg(
    pl.len().alias("total_flights"),
    pl.col("distance").mean().alias("average_dist")
).sort("total_flights", descending=True)

# --------------------------
# Joins
# --------------------------

# replace airlines code with their name
carriers_info = pl.read_csv("data/pa-flights/airlines.csv", null_values=["NA", ""])
carriers = carriers.join(carriers_info, on="carrier", how="inner").select(["name", "avg_dep_delay"])

# join with airports for destination city and timezone
airports_info = pl.read_csv("data/pa-flights/airports.csv", null_values=["NA", ""]).select(["faa", "name", "tzone"])

airports_info = airports_info.rename({"name": "name_origin", "tzone": "tzone_origin"})
flights = flights.join(airports_info, left_on="origin", right_on="faa", how="left")

airports_info = airports_info.rename({"name_origin": "name_dest", "tzone_origin": "tzone_dest"})
flights = flights.join(airports_info, left_on="dest", right_on="faa", how="left")

# --------------------------
# Pivots
# --------------------------
departure_delays = flights.pivot("month", index="carrier", values="dep_delay", aggregate_function="mean")

# --------------------------
# Answering Questions
# --------------------------

# Question I am trying to answer:
# Which time of the year is better to fly to the west, accounting for weather at departure, from Pittsburgh?

flights = pl.read_csv(path_flights, infer_schema_length=10000, null_values=["NA", ""])
airports_info = pl.read_csv("data/pa-flights/airports.csv", null_values=["NA", ""])
weather = pl.read_csv("data/pa-flights/weather.csv",  null_values=["NA", ""]).filter(pl.col("origin") == "PIT")

# To account for severe weather, I take the extreme events for each weather type using quantiles
weather = weather.with_columns(
    (pl.col("precip") >= pl.col("precip").quantile(0.95)).alias("severe_precip"),
    (pl.col("wind_gust") >= pl.col("wind_gust").quantile(0.95)).alias("severe_gust"),
    (pl.col("visib") <= pl.col("visib").quantile(0.05)).alias("low_vis")
).select(["origin", "year", "month", "day", "hour", "severe_precip", "severe_gust", "low_vis"])

pit_west = (flights.filter(pl.col("origin") == "PIT"
    ).join(airports_info, left_on="dest", right_on="faa", how="left"
    ).filter(pl.col("lon") < -100
    ).join(weather, on=["origin", "year", "month", "day", "hour"], how="left")
    .group_by("month").agg(
        pl.len().alias("n"),
        (pl.col("arr_delay") >= 15).mean().alias("frac_delays"),
        (pl.col("arr_delay") >= 15).filter(pl.col("severe_precip")).mean().alias("frac_delays_precip"),
        (pl.col("arr_delay") >= 15).filter(pl.col("severe_gust")).mean().alias("frac_delays_gust"),
        (pl.col("arr_delay") >= 15).filter(pl.col("low_vis")).mean().alias("frac_delays_vis"),
        pl.col("severe_precip").sum().alias("n_precip"),
        pl.col("severe_gust").sum().alias("n_gust"),
        pl.col("low_vis").sum().alias("n_vis"),
    ).sort("month"))