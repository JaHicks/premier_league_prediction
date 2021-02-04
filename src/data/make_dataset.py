# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from datetime import datetime
import pandas as pd
import shutil
import urllib.request


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('interim_filepath', type=click.Path())
def main(input_filepath, interim_filepath):
    """ Runs data processing scripts to download raw data and process it
         with interim steps saved along the way.
    """
    logger = logging.getLogger(__name__)

    # TODO: number the interim files so the order is clearer

    # downloads CSVs to data/raw
    download_match_data(input_filepath)

    # generates {interim_filepath}/matches_raw.csv
    standardise_data_format(input_filepath, interim_filepath)

    # generates {interim_filepath}/matches_imputed_missing.csv
    impute_missing_values(interim_filepath)

    # generates {interim_filepath}/matches_corrected.csv
    correct_data_entry_mistakes(interim_filepath)

    # generates {interim_filepath}/matches_expanded_features.csv
    expand_for_against_values(interim_filepath)

    logger.info("Downloaded and processed data")


def download_match_data(download_directory, leagues=["E"],
                        seasons=range(2000, 2021), divisions=range(2)):
    """ Downloads raw match data CSVs from football-data.co.uk
    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading data...")
    url_base = "http://www.football-data.co.uk/mmz4281"
    for league in leagues:
        for season in seasons:
            for division in divisions:
                # Generate URL and filename for each season and division
                season_start_year = str(season)[2:]
                season_end_year = str(season+1)[2:]
                season_string = "{}{}".format(season_start_year,
                                              season_end_year)

                url = "{}/{}/{}{}.csv".format(url_base, season_string,
                                              league, division)
                filename = "{}/{}-{}{}.csv".format(download_directory, season,
                                                   league, division)

                # Download CSVs
                with urllib.request.urlopen(url) as response, \
                        open(filename, 'wb') as outfile:
                    shutil.copyfileobj(response, outfile)
                    logger.info("Downloaded file: {}".format(filename))


def standardise_data_format(download_directory, interim_filepath,
                            leagues=["E"], seasons=range(2000, 2021),
                            divisions=range(2)):
    """ Combines all matches in to single CSV with human readable column names
        and standardised date format
    """
    logger = logging.getLogger(__name__)
    logger.info("Standardising data...")
    # Data files have slight mixture of columns used
    #  so specify which ones we want to use
    usecols = ["Date", "HomeTeam", "AwayTeam", "FTR",
               "FTHG", "HS", "HST", "HC", "HF", "HY", "HR",
               "FTAG", "AS", "AST", "AC", "AF", "AY", "AR",
               "IWH", "IWD", "IWA", "WHH", "WHD", "WHA"]

    # Human readable column names
    rename_columns = {
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTR": "result",
        "FTHG": "home_goals",
        "HS": "home_shots",
        "HST": "home_shotsOnTarget",
        "HC": "home_corners",
        "HF": "home_fouls",
        "HY": "home_yellowCards",
        "HR": "home_redCards",
        "FTAG": "away_goals",
        "AS": "away_shots",
        "AST": "away_shotsOnTarget",
        "AC": "away_corners",
        "AF": "away_fouls",
        "AY": "away_yellowCards",
        "AR": "away_redCards",
        # Betting odds in case we want to compare our predictions later
        "IWH": "odds_interwetten_homeWin",
        "IWD": "odds_interwetten_draw",
        "IWA": "odds_interwetten_awayWin",
        "WHH": "odds_williamHill_homeWin",
        "WHD": "odds_williamHill_draw",
        "WHA": "odds_williamHill_awayWin",
    }

    # New empty dataframe to append records to
    matches = pd.DataFrame()

    for league in leagues:
        for season in seasons:
            for division in divisions:
                filename = "{}/{}-{}{}.csv".format(download_directory, season,
                                                   league, division)
                logger.info("Loading: {}".format(filename))
                df = pd.read_csv(filename, usecols=usecols)

                # Drop empty rows (e.g. at the end of the file)
                df.dropna(how="all", inplace=True)

                # Make existing columns human readable
                df.rename(columns=rename_columns, inplace=True)

                # Standardise date fromat
                # Files have mixture of 2 different date formats, so try one
                #  then use the other if the first fails
                try:
                    date_format = "%d/%m/%y"
                    df["date"] = df["date"].apply(
                        lambda x: datetime.strptime(x, date_format).date())
                except ValueError:
                    date_format = "%d/%m/%Y"
                    df["date"] = df["date"].apply(
                        lambda x: datetime.strptime(x, date_format).date())

                # Retain data about league, season and division
                df["league"] = league  # "E" for English
                df["season"] = season  # year of start of season
                if division == 0:
                    df["division"] = "premier"
                elif division == 1:
                    df["division"] = "championship"
                else:
                    # Fallback if using more than the first 2 divisions
                    df["division"] = "div_{}".format(division)

                # Append to single dataframe containing all matches
                matches = matches.append(df, ignore_index=True)

    # Ensure matches sorted by date
    matches.sort_values(by="date", ignore_index=True, inplace=True)

    # Save formatted raw match data in one file
    filename = "{}/matches_raw.csv".format(interim_filepath)
    matches.to_csv(filename, index=False)
    logger.info("Saved file: {}".format(filename))


def impute_missing_values(interim_filepath):
    """ Use medians of all previous values from same division
         with same number of home goals and away goals
    """
    logger = logging.getLogger(__name__)
    logger.info("Imputing missing values...")

    matches = pd.read_csv("{}/matches_raw.csv".format(interim_filepath))

    numeric_features = matches.dtypes[matches.dtypes != "object"].index
    features_to_exclude_from_imputing = [
        "league",
        "season",
        "division",
        "home_goals",
        "away_goals",
        "odds_interwetten_homeWin",
        "odds_interwetten_draw",
        "odds_interwetten_awayWin",
        "odds_williamHill_homeWin",
        "odds_williamHill_draw",
        "odds_williamHill_awayWin"
    ]

    for feature in numeric_features:
        if feature not in features_to_exclude_from_imputing:

            groupby_cols = ["league", "division", "home_goals", "away_goals"]
            matches[feature] = matches.groupby(groupby_cols)[feature].\
                transform(lambda x: x.fillna(x.shift(1).expanding().median()))

    # Save matches with imputed values
    filename = "{}/matches_imputed_missing.csv".format(interim_filepath)
    matches.to_csv(filename, index=False)
    logger.info("Saved file: {}".format(filename))

    logger.info("Completed imputing missing values!")


def correct_data_entry_mistakes(interim_filepath):
    """ Add 10 to shots where the shots are less than shots on target
        These were most likely data entry mistakes with a missing 1
    """
    logger = logging.getLogger(__name__)
    logger.info("Correcting data entry mistakes...")

    matches = pd.read_csv("{}/matches_imputed_missing.csv".format(
        interim_filepath))

    matches.loc[matches["home_shotsOnTarget"] > matches["home_shots"],
                "home_shots"] += 10
    matches.loc[matches["away_shotsOnTarget"] > matches["away_shots"],
                "away_shots"] += 10

    # Save matches with corrected values
    filename = "{}/matches_corrected.csv".format(interim_filepath)
    matches.to_csv(filename, index=False)
    logger.info("Saved file: {}".format(filename))

    logger.info("Completed correcting data entry mistakes!")


def expand_for_against_values(interim_filepath):
    """ Create for/against features (makes manipulation easier later) - e.g.
         home_goals => home_goals_for = away_goals_against
         away_goals => away_goals_for = home_goals_against
    """
    logger = logging.getLogger(__name__)
    logger.info("Expanding for and against features...")

    matches = pd.read_csv("{}/matches_corrected.csv".format(
        interim_filepath))

    feature_bases = ["goals", "shots", "shotsOnTarget", "corners", "fouls",
                     "yellowCards", "redCards"]
    for feature_base in feature_bases:
        home_feature = "home_{}".format(feature_base)
        away_feature = "away_{}".format(feature_base)

        home_for_feature = "{}_for".format(home_feature)
        home_against_feature = "{}_against".format(home_feature)
        away_for_feature = "{}_for".format(away_feature)
        away_against_feature = "{}_against".format(away_feature)

        matches[home_for_feature] = matches[home_feature]
        matches[home_against_feature] = matches[away_feature]
        matches[away_for_feature] = matches[away_feature]
        matches[away_against_feature] = matches[home_feature]

        matches.drop(columns=[home_feature, away_feature], inplace=True)

    # Save matches with for and against values
    filename = "{}/matches_expanded_features.csv".format(interim_filepath)
    matches.to_csv(filename, index=False)
    logger.info("Saved file: {}".format(filename))

    logger.info("Expanded for and against features!")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
