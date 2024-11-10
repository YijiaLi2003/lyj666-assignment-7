from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from flask_session import Session

app = Flask(__name__)
app.secret_key = "1"  # Replace with your own secret key

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    error = np.random.normal(0, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + mu + error

    # Fit a linear regression model to X and Y
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure()
    plt.scatter(X, Y, label='Data Points')
    plt.plot(X, model.predict(X_reshaped), color='red', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Regression Line: Y = {slope:.2f}X + {intercept:.2f}')
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.rand(N)
        error_sim = np.random.normal(0, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + mu + error_sim

        # Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression()
        X_sim_reshaped = X_sim.reshape(-1, 1)
        sim_model.fit(X_sim_reshaped, Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Observed Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Observed Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i > intercept for i in intercepts) / S

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_more_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = session.get("N")
    S = session.get("S")
    slope = session.get("slope")
    intercept = session.get("intercept")
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = session.get("beta0")
    beta1 = session.get("beta1")

    # Ensure all necessary session data is available
    if None in (N, S, slope, intercept, slopes, intercepts, beta0, beta1):
        return "Session data is missing. Please generate data first.", 400

    N = int(N)
    S = int(S)
    slope = float(slope)
    intercept = float(intercept)
    beta0 = float(beta0)
    beta1 = float(beta1)

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Map test_type to expected values
    test_type_mapping = {
        '>': 'greater',
        '<': 'less',
        '!=': 'not_equal'
    }

    if test_type not in test_type_mapping:
        return "Invalid test type selected. Please choose a valid test type.", 400

    mapped_test_type = test_type_mapping[test_type]

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on mapped test type
    if mapped_test_type == 'greater':
        p_value = np.sum(simulated_stats >= observed_stat) / S
    elif mapped_test_type == 'less':
        p_value = np.sum(simulated_stats <= observed_stat) / S
    elif mapped_test_type == 'not_equal':
        p_value = np.sum(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value)) / S

    # If p_value is very small, set fun_message
    if p_value is not None and p_value <= 0.0001:
        fun_message = "Wow! That's a tiny p-value! You've encountered a rare event!"
    else:
        fun_message = None

    # Plot histogram of simulated statistics
    plt.figure()
    plt.hist(simulated_stats, bins=30, alpha=0.7, label='Simulated Statistics')
    plt.axvline(observed_stat, color='red', linestyle='--', linewidth=2, label=f'Observed {parameter}: {observed_stat:.2f}')
    plt.axvline(hypothesized_value, color='green', linestyle='-', linewidth=2, label=f'Hypothesized {parameter}: {hypothesized_value:.2f}')
    plt.xlabel(f'{parameter.capitalize()} Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Simulated Statistics')
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )


@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = session.get("N")
    mu = session.get("mu")
    sigma2 = session.get("sigma2")
    beta0 = session.get("beta0")
    beta1 = session.get("beta1")
    S = session.get("S")
    slope = session.get("slope")
    intercept = session.get("intercept")
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    # Ensure all necessary session data is available
    if None in (N, mu, sigma2, beta0, beta1, S, slope, intercept, slopes, intercepts):
        return "Session data is missing. Please generate data first.", 400

    N = int(N)
    mu = float(mu)
    sigma2 = float(sigma2)
    beta0 = float(beta0)
    beta1 = float(beta1)
    S = int(S)
    slope = float(slope)
    intercept = float(intercept)

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    # Calculate confidence interval for the parameter estimate
    alpha = 1 - confidence_level / 100
    df = S - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    se_estimate = std_estimate / np.sqrt(S)
    ci_lower = mean_estimate - t_crit * se_estimate
    ci_upper = mean_estimate + t_crit * se_estimate

    # Check if confidence interval includes true parameter
    includes_true = (ci_lower <= true_param) and (ci_upper >= true_param)

    # Plot the individual estimates as gray points and confidence interval
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(estimates)), estimates, color='gray', alpha=0.5, label='Simulated Estimates')

    # Plot the mean estimate
    mean_color = 'green' if includes_true else 'red'
    plt.scatter(S + 1, mean_estimate, color=mean_color, label=f'Mean Estimate: {mean_estimate:.2f}')

    # Plot the confidence interval
    plt.hlines(mean_estimate, xmin=0, xmax=S, colors=mean_color, linestyles='-', lw=2)
    plt.fill_between([0, S], ci_lower, ci_upper, color=mean_color, alpha=0.2, label=f'{confidence_level}% Confidence Interval')

    # Plot the true parameter value
    plt.axhline(y=true_param, color='blue', linestyle='--', label=f'True {parameter.capitalize()}: {true_param:.2f}')

    plt.xlabel('Simulation Index')
    plt.ylabel(f'{parameter.capitalize()} Estimates')
    plt.title(f'Confidence Interval for {parameter.capitalize()}')
    plt.legend()
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)
