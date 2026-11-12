from scipy.stats import norm

print("================================")
print("       NFL HANDICAPPING MODEL")
print("================================")

# Enter teams
home = input("Home team: ")
away = input("Away team: ")

# Enter team power ratings
home_rating = float(input("Home team power rating: "))
away_rating = float(input("Away team power rating: "))

# Home field advantage
home_advantage = 1.5

# Calculate projected margin
projected_margin = (
    home_rating - away_rating + home_advantage
)

# Calculate win probabilities
home_win_probability = norm.cdf(
    projected_margin / 13.5
)

away_win_probability = 1 - home_win_probability

# Determine predicted winner
if projected_margin > 0:
    predicted_winner = home
elif projected_margin < 0:
    predicted_winner = away
else:
    predicted_winner = "Too close to call"

# Display results
print("\n================================")
print("          MODEL RESULTS")
print("================================")

print(f"Matchup: {away} at {home}")
print(f"Projected home margin: {projected_margin:+.1f}")
print(f"Home win probability: {home_win_probability:.1%}")
print(f"Away win probability: {away_win_probability:.1%}")
print(f"Predicted winner: {predicted_winner}")

print("================================")

from scipy.stats import norm

print("================================")
print("       NFL HANDICAPPING MODEL")
print("================================")

# Enter teams
home = input("Home team: ")
away = input("Away team: ")

# Enter team power ratings
home_rating = float(input("Home team power rating: "))
away_rating = float(input("Away team power rating: "))

# Home field advantage
home_advantage = 1.5

# Calculate projected margin
projected_margin = (
    home_rating - away_rating + home_advantage
)

# Calculate win probabilities
home_win_probability = norm.cdf(
    projected_margin / 13.5
)

away_win_probability = 1 - home_win_probability

# Determine predicted winner
if projected_margin > 0:
    predicted_winner = home
elif projected_margin < 0:
    predicted_winner = away
else:
    predicted_winner = "Too close to call"

# Display results
print("\n================================")
print("          MODEL RESULTS")
print("================================")

print(f"Matchup: {away} at {home}")
print(f"Projected home margin: {projected_margin:+.1f}")
print(f"Home win probability: {home_win_probability:.1%}")
print(f"Away win probability: {away_win_probability:.1%}")
print(f"Predicted winner: {predicted_winner}")

print("================================")
