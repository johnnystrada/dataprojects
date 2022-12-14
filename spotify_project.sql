#opening statistics
Use spotify_covid;

#Spotify statistics on the top 10 streamed songs with #1 chart ranks
select
s.songID,
s.songTitle,
a.artistName,
MIN(sc.streams) as minStreams,
MAX(sc.streams) as maxStreams,
ROUND(AVG(sc.streams)) as avgStreams,
SUM(sc.streams) as totalStreams,
ROUND(STD(sc.streams)) as stdStreams
FROM
spotify_charts sc
inner join
songs s on s.songID = sc.songID
inner join
artists a on a.artistID = s.artistID
where
sc.rankNum = 1
group by s.songID
order by totalStreams DESC
limit 10; 


# Total song count in the spotify dataset ordered descending by song count
SELECT
songTitle 'Song Title',
count(songTitle) as 'Song Count',
artistName 'Artist'
FROM
    spotify_charts sc
inner join songs ss on sc.songID = ss.songID
inner join artists sa on ss.artistID = sa.artistID
GROUP BY songTitle
ORDER BY COUNT(songTitle) DESC
limit 10;

# Total artist count in the spotify dataset ordered descending by artist count
SELECT
artistName 'Artist',
count(artistName) as 'Artist Count'
FROM
    spotify_charts sc
inner join songs ss on sc.songID = ss.songID
inner join artists sa on ss.artistID = sa.artistID
Group By artistName Order By count(artistName) DESC
limit 10;


#Total genre count in the spotify dataset ordered descending by genre count
Select genre,
count(genre) as 'Genre Count'
from spotify_charts sc
inner join songs ss on sc.songID = ss.songID
inner join genre sg on ss.genreID = sg.genreID
GROUP By genre ORDER BY count(genre) DESC
limit 10;


#Top 10 Artists when change in covid hospital total was highest
SELECT
sa.artistName as 'Artist',
songTitle 'Song Title',
genre as 'Genre',
rankNum as 'Chart Rank',
sc.streams as ' Total Streams',
sc.date
FROM
    spotify_charts sc
inner join songs ss on sc.songID = ss.songID
inner join artists sa on ss.artistID = sa.artistID
inner join genre sg on ss.genreID = sg.genreID
inner join dates on sc.date = dates.date
where sc.date =
(
SELECT 
end_of_week
from covid_hospital ch order by cumulative_rate_change DESC LIMIT 1
)
 GROUP BY songTitle ORDER BY sc.rankNum ASC LIMIT 10;
 
 
 #Top 10 Artists when total covid hospitalizations increase was highest in 2021
SELECT
sa.artistName as 'Artist',
songTitle 'Song Title',
genre as 'Genre',
rankNum as 'Chart Rank',
sc.streams as ' Total Streams',
sc.date
FROM
    spotify_charts sc
inner join songs ss on sc.songID = ss.songID
inner join artists sa on ss.artistID = sa.artistID
inner join genre sg on ss.genreID = sg.genreID
inner join dates on sc.date = dates.date
where sc.date =
(
SELECT 
end_of_week
from covid_hospital ch where end_of_week > '2020-12-31' order by cumulative_rate_change DESC LIMIT 1
)
 GROUP BY songTitle ORDER BY sc.rankNum ASC LIMIT 10;


#Highest ranking song during greatest week to week rate of change during the pandemic
select
s.songTitle, a.artistName, c.year, c.end_of_week, c.weekly_change_rate
from covid_hospital c 
join dates d on d.endOfWeek = c.end_of_week
join spotify_charts sc on d.date = sc.date
INNER JOIN songs s ON s.songID = sc.songID
INNER join
artists a on a.artistID = s.artistID
where sc.rankNum = 1
group by songTitle
Order by c.year, c.weekly_change_rate DESC
limit 10;


#Highest ranking genre during greatest week to week change in covid hospitalizations
select g.genre, c.year, c.end_of_week, c.weekly_change_total
from covid_hospital c 
join dates d on d.endOfWeek = c.end_of_week
join spotify_charts sc on d.date = sc.date
INNER JOIN songs s ON s.songID = sc.songID
join genre g on g.genreID = s.genreID
group by g.genre
Order by c.weekly_change_total DESC
limit 10;


 #Total Streams when cumulative Covid Change was the highest versus Total Streams When cumulative Covid change was lowest
SELECT
ch.cumulative_change_total as 'Covid Change Total',
 sum(streams) as 'Total Streams',
 sc.date as 'Date'
 #sc.date
 FROM
    spotify_charts sc
    inner join covid_hospital ch on sc.date = ch.end_of_week
    where sc.date =
(
SELECT 
end_of_week
from covid_hospital ch where end_of_week < '2022-01-01' order by cumulative_change_total DESC LIMIT 1
) 
UNION 
 #Total Streams when Covid Change was the lowest
SELECT
cumulative_change_total,
 sum(streams) as 'Total Streams',
 sc.date as 'Date'
 #sc.date
 FROM
    spotify_charts sc
    inner join covid_hospital ch on sc.date = ch.end_of_week
    where sc.date =
(
SELECT 
end_of_week
from covid_hospital ch where end_of_week < '2022-01-01'
order by
cumulative_change_total ASC 
LIMIT 1
);

 #Total Top Artist & Streams when Covid Change was the highest/lowest
SELECT
ch.cumulative_change_total as 'Covid Change Total',
 sa.artistName as 'Top Chart Artist',
 sum(sc.streams) as 'Total Streams',
 sc.date as 'Date'
 #sc.date
 FROM
    spotify_charts sc
    inner join covid_hospital ch on sc.date = ch.end_of_week
    inner join songs ss on sc.songID = ss.songID
    inner join artists sa on ss.artistId = sa.artistId
    where sc.rankNum = 1 AND sc.date =
(
SELECT 
end_of_week
from covid_hospital ch where end_of_week < '2022-01-01' order by cumulative_change_total DESC LIMIT 1
)
UNION 
 #Total Streams when Covid Change was the lowest
SELECT
cumulative_change_total,
sa.artistName as 'Top Chart Artist',
sum(sc.streams) as 'Total Streams',
sc.date as 'Date'
FROM
    spotify_charts sc
	inner join covid_hospital ch on sc.date = ch.end_of_week
    inner join songs ss on sc.songID = ss.songID
    inner join artists sa on ss.artistId = sa.artistId
    where sc.rankNum = 1 AND sc.date =
(
SELECT 
end_of_week
from covid_hospital ch where end_of_week < '2022-01-01'
order by
cumulative_change_total ASC 
LIMIT 1
);


### Average characteristics of Top 200 Spotify songs by week, along with average COVID hospitalizations for that same week ###
select d.endofweek, ch.weekly_rate AS weeklyCovidRate, avg(s.valence) AS averageValence,
					avg(s.tempo) AS averageTempo, avg(s.energy) AS averageEnergy
from songs s
INNER JOIN spotify_charts sc ON s.songID = sc.songID
INNER JOIN dates d ON sc.date = d.date
LEFT OUTER JOIN covid_hospital ch ON ch.end_of_week = d.endOfWeek
where 
ch.end_of_week is not null
GROUP BY d.endofweek
ORDER BY d.endofweek;

#Each artists' song with its highest charted position and week
SELECT 
    sa.artistName AS 'Artist',
    songTitle 'Song Title',
    genre AS 'Genre',
    rankNum AS 'Chart Rank',
    sc.streams AS ' Total Streams',
    dates.endOfWeek
FROM
    spotify_charts sc
        INNER JOIN
    songs ss ON sc.songID = ss.songID
        INNER JOIN
    artists sa ON ss.artistID = sa.artistID
        INNER JOIN
    genre sg ON ss.genreID = sg.genreID
        INNER JOIN
    dates ON sc.date = dates.date
GROUP BY sa.artistName , ss.songTitle
ORDER BY sc.rankNum;

#Covid Hospitalized Weekly and Cumulative Total
SELECT 
    cv.end_of_week,
    weekly_hospital_total as 'Week Hospitalized Total',
    cumulative_hospital_total as 'Cumulative Hospitalized Total'
FROM
    covid_hospital cv
ORDER BY end_of_week ASC;
