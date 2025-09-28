Data-Driven Financial Planning System for Elips Medikal (Elips Sağlık Ürünleri)
Introduction and Overview

Elips Medikal (Elips Sağlık Ürünleri) is a Turkish life-sciences company that specializes in importing and distributing advanced laboratory and medical equipment. The firm provides innovative solutions in the life sciences field and operates in import/export with a strong customer network across Turkey, connecting researchers and healthcare professionals with the latest high-quality lab products
elipsltd.com.tr
. Founded in 1999
emis.com
, Elips Medikal focuses on sales and after-sales support of molecular genetics and biotech laboratory tools – from genomic analysis devices and real-time PCR machines to automatic pipetting robots, centrifuges, and spectrophotometers
emis.com
. The company even established an R&D lab (“Ibn-i Sina”) to offer COVID-19 related products and services during the pandemic
emis.com
, reflecting its commitment to innovation.

Financial Planning Need: As a growing distributor of scientific equipment, Elips Medikal faces complex financial planning challenges. Importing high-tech devices means dealing with foreign currencies, fluctuating costs, and inventory management, while serving hospitals, labs, and universities often involves irregular procurement cycles (e.g. large tender orders) and credit terms. Effective financial planning is critical – the company must forecast sales revenue, manage expenses, and plan budgets in a volatile market (especially with currency fluctuations). Currently, planning might rely on manual spreadsheets and intuition. Adopting a data-driven approach can greatly enhance accuracy and strategic decision-making. Modern Financial Planning & Analysis (FP&A) practices leverage data science techniques to detect patterns in historical data and produce predictive models for forecasting future performance
rtslabs.com
. They also enable instant “what-if” scenario analysis, allowing finance teams to evaluate decisions under different assumptions (for example, testing how changes in pricing or exchange rates affect profits)
rtslabs.com
.

Project Goal: The goal is to develop a Python-based financial planning system for Elips Medikal that integrates data science and finance. This system will use the company’s historical data (or well-chosen sample data, given limited internal access) to forecast key financial metrics (like sales, revenues, and cash flows) and to perform scenario analysis for strategic planning. By combining predictive analytics with financial modeling, the tool will help Elips Medikal’s management make informed decisions, optimize budgets, and anticipate risks. The solution will be implemented as a command-line interface (CLI) application in Python – meaning it can run in a console environment, taking inputs (like data files or scenario parameters) and outputting analysis results in text and charts. A CLI tool is sufficient (no need for a fancy web UI) and keeps the focus on functionality and execution in a local environment, which is suitable for both the company’s practical use and demonstrating technical mastery in a data science/finance graduate program setting.

Benefits: This project is beneficial to the company and academically noteworthy. For Elips Medikal, it provides a data-driven FP&A system that improves forecasting accuracy, saves time on manual analyses, and allows exploring multiple scenarios quickly (e.g. best-case, worst-case, currency fluctuations) to choose optimal strategies. In fact, AI-driven planning systems can model real-time scenarios and test many “what-if” cases instantly, saving finance teams significant effort
rtslabs.com
. The system will help identify trends (e.g. seasonal demand for certain lab products), optimize inventory and purchasing by anticipating needs, and manage financial risks (like exchange rate impact on costs) proactively. For a master’s program or academic interest, this project is compelling because it sits at the intersection of data science and finance – demonstrating how machine learning and statistical modeling can be applied to corporate financial planning. It showcases skills in data analysis, time-series forecasting, scenario simulation, and software development, all in one project. The end result will be a tangible, working CLI program in Python, which is executable step-by-step and can be further extended, making it an ideal capstone or showcase project that is both practical and innovative.

System Overview and Design

Proposed Solution: The financial planning system will consist of several components working together as a pipeline, implemented in Python:

Data Input & Storage: Historical financial and operational data will feed the system. This includes past sales figures (e.g. monthly or quarterly revenue by product category), cost of goods sold, operating expenses, and any other relevant metrics (like number of units sold, or external data such as exchange rates). Since we cannot access real internal databases, we will use either publicly available data (if any) or simulated data representative of Elips Medikal’s business. The data can be stored in CSV files or spreadsheets which the Python program will read. We assume the structure might include date columns, product or category, sales amount, etc. For external factors, we might prepare a file for, say, monthly USD/TRY exchange rate if needed for scenario analysis.

Data Processing & Analysis Module: Using Python’s data science libraries (like pandas for data manipulation and NumPy for calculations), the system will clean and preprocess the data. This involves parsing dates, handling missing values, and organizing the information into time series (e.g. total monthly sales). An exploratory analysis will be done to understand trends and patterns: for example, identifying peak sales periods, growth trends, or correlations (does a particular product line drive most revenue? how does currency rate correlate with profit margins?). This helps in choosing the right forecasting approach. We will also split data into training vs. testing periods to validate our models.

Forecasting Model: At the core of the system is a predictive model that generates forecasts for future periods. We will likely implement a time series forecasting approach for revenues (and possibly for major expense categories). A classical method is to use an ARIMA model (Auto-Regressive Integrated Moving Average) or exponential smoothing for time series, which is well-supported by Python’s statsmodels or pmdarima library. These models can capture trend and seasonality in Elips’s sales data (for instance, if sales spike at year-end or during certain months). We can also consider Facebook Prophet (a modern library for time series forecasting) which automatically handles seasonality and holidays – this might be useful if, say, research institutes typically make purchases at fiscal year-end or if there are seasonal budget cycles. The model will be trained on historical data to predict future sales. We will evaluate its accuracy using metrics like MAPE (Mean Absolute Percentage Error) on a hold-out sample of historical data. If needed, we may experiment with more complex models (like a regression or machine learning model that includes external variables such as macroeconomic indicators or exchange rates as features), but given the scope and the desire to keep it doable and not overly complex, an ARIMA/Prophet approach for sales forecasting should suffice and yield interpretable results.

Financial Planning Module: This module takes the forecasted values and turns them into a usable financial plan. Essentially, it will produce a projected income statement (at least the revenue and gross profit, potentially also operating profit) for upcoming periods. For example, using the forecasted sales, the system can compute expected revenue. Then, using assumptions about cost of goods sold (perhaps a percentage of sales or specific margin by product category), it can estimate the cost and gross profit. Operating expenses (salaries, rent, etc.) might be forecasted by simple methods (e.g. assume they grow at a steady rate or use the average of last few periods, unless we have data to model them). The output could be a budget for the next year or quarter, broken down by period. Importantly, this module will allow scenario adjustments: the user (via CLI inputs or configuration) can modify key assumptions and see the impact. For instance, what if the currency exchange rate is expected to change by 10%? The system could adjust the cost of imported goods accordingly and show the new profit forecast. Or what if a new product line is introduced, increasing sales by X%? We could incorporate that as an uplift in the forecast. The idea is to enable dynamic “what-if” analysis similar to how AI-driven FP&A tools let CFOs compare different strategies under various conditions
rtslabs.com
. This scenario capability makes the planning more robust and insightful than a single static forecast.

User Interface (CLI): The program will be operated through a command-line interface. This means a user (e.g. a financial analyst at Elips or a student running the project) will interact by running the Python script with certain commands or prompts. We will design it to be user-friendly in text form. For example, one might run python plan_system.py --forecast 12 to forecast 12 months ahead, or enter interactive mode where the script asks for input like “Enter scenario: (1) Base case, (2) High sales, (3) FX shock”. The CLI will then execute the appropriate analysis and output results to the terminal and/or to output files. Results could include summary tables of projected financials and even charts (for example, a line chart of past vs. forecasted sales) saved as an image file for review. The use of a CLI ensures focus on functionality – since the user specifically did not require a web or GUI application, we keep things simple and transparent. It also makes the system easier to run on any machine without special software (just Python environment).

Python Focus: All components will be implemented in Python, leveraging its rich ecosystem for data science and finance. Key libraries likely to be used include: pandas for data cleaning and financial calculations, statsmodels or prophet for time series forecasting, matplotlib or plotly for generating any visualizations (if needed for understanding trends or presenting results), and Python’s standard libraries (csv or openpyxl) for reading data files. We might also use numpy and possibly scikit-learn if any machine learning or additional regressions are applied. The CLI aspect can be handled simply by argparse (for command-line arguments) or an interactive loop with input() prompts for the user. The overall code will be structured to allow step-by-step execution and easy tweaking (for academic demonstration purposes, one could run each step in a Jupyter notebook as well, but the final deliverable will run in terminal).

In summary, the system will take Elips Medikal’s historical financial data, analyze and model it to forecast future outcomes, and produce a financial plan with the ability to run different scenarios. It’s essentially like building a smart budgeting tool tailored to the company’s needs, using data science techniques to improve on traditional spreadsheet planning. Next, we outline the detailed steps to build this system from scratch to completion.

Step-by-Step Implementation Plan

Below is a step-by-step plan for developing the data science & finance integrated planning system for Elips Medikal. Each step is described in detail to ensure clarity. By following these steps in order, we will progress from initial research through development, resulting in a functional Python CLI tool for financial planning. (Note: Since we lack actual internal data, some steps involve creating or assuming sample data. The plan remains feasible and focused – it avoids unnecessary complexity while still achieving a robust solution.)

Define Project Scope and Requirements: Begin by confirming the scope and objectives of the project. Clearly document what financial questions the system should answer for Elips Medikal. For example: “Forecast next 12 months of sales and profit”, “Allow scenario analysis for exchange rate changes”, “Provide budget recommendations for spending.” Identify the key stakeholders (e.g. the company’s finance manager or CFO) and what metrics matter most to them (revenue, gross profit, cash flow, etc.). Since we know financial planning is the priority, ensure the requirements emphasize accurate forecasting and budgeting capabilities. Also note technical requirements: the solution will be Python-based and run as a CLI tool (so no need for GUI or web aspects). Defining the scope up front helps keep the project on track and not overly complex. It also sets success criteria (e.g., forecast error below X%, ability to run at least 3 different scenarios, etc.).

Gather Data and Assumptions: Assemble all relevant data needed for analysis and modeling. Ideally, this includes historical financial data from Elips Medikal: past sales revenue by month/quarter, broken down by product category if possible; historical costs (cost of goods sold, operating expenses); and any other operational data (like number of units sold, or number of service contracts). In practice, because we cannot fetch the company’s internal records directly, we will simulate or use proxy data:

Sales Data: Create a synthetic dataset of monthly sales for, say, the past 3–5 years. This can be done by using known industry patterns or reasonable assumptions. For instance, assume steady growth with some seasonality. (If any public info is available – e.g. a mention of their import volumes or market share – use it to calibrate the magnitude. The company deals with costly lab devices, so sales might be lumpy; perhaps a few big deals per quarter. We might introduce random spikes to simulate that.)

Expense Data: Assume or gather typical expense ratios. If Elips Medikal is a distributor, cost of goods sold could be, say, 60–70% of sales (since they import devices for resale). Operating expenses (salaries, rent, marketing) might be relatively fixed – we can model them as a fixed amount per month plus some inflation. If we find any industry benchmarks, we’ll use them; if not, we’ll set reasonable values (ensuring the planning output looks realistic).

External Data: Download or generate any external factors needed. A crucial one is the USD/TRY exchange rate over the same historical period, since fluctuations in the Turkish Lira can significantly affect import costs and pricing. This data can be obtained from public sources (e.g. central bank or finance APIs) or manually input approximate figures. Also consider inflation rates or any major economic events in the timeline (for scenario context, e.g. a big jump in 2018 or 2021 for TRY).

Master Data Structure: Organize the data in files (CSV or Excel). For example, sales_history.csv with columns: Date, Sales_TRY. If multiple product categories, include those as separate columns or a category column. Another file for expenses.csv (or we integrate expenses in the same file by columns: COGS, Opex, etc., per period). Ensure the data covers a continuous timeline with consistent intervals (monthly is recommended for resolution).

Assumption Documentation: Alongside the raw numbers, keep a note of what each data series represents and any assumptions (for transparency). This will help later in scenario analysis (e.g., if we assume 65% gross margin, note that so we can adjust it in a scenario if needed).

Set Up the Python Environment: Before heavy analysis, set up your development environment with necessary tools and libraries. Install Python 3.x and ensure you have access to a terminal/command line. Create a project folder (e.g. ElipsFinancialPlanner/). Use a virtual environment if needed. Install key libraries:

pandas (for data handling),

numpy (numerical computations),

matplotlib or seaborn (for plotting, optional but useful for EDA or output visualization),

statsmodels (for ARIMA and other statistical models) and/or pmdarima (which has an auto-ARIMA that can find optimal parameters),

prophet (optional, for an alternative forecasting method; this is prophet by Facebook, might require pip install prophet and also installing pystan/ cmdstanpy depending on environment),

scikit-learn (if using any regression or machine learning model for comparison),

argparse (for building the CLI interface to parse command-line options).
Ensure everything installs properly and test by importing these in a Python REPL. Setting up early avoids issues later when integrating components.

Data Import and Cleaning (Coding Step): Write Python code to load the datasets prepared in Step 2. Using pandas, read the CSV files into DataFrame objects. Perform data cleaning:

Parse dates into proper datetime objects (e.g. if using pandas.read_csv, specify parse_dates=[DateColumn]).

Sort data by date and ensure there are no gaps in the timeline. If a month is missing (no sales recorded, perhaps zero), consider filling it with 0 or interpolating if appropriate, but typically in financial data we’d explicitly handle zero-sales months if any.

Handle missing values or outliers: e.g., if a value is extreme, check if it’s plausible (maybe a big tender landed that month – if plausible, keep it; if it looks like an error, you might smooth it or mark it). Since data is synthetic or from assumed sources, we ensure consistency ourselves.

Add any computed fields that might help analysis, like calculating gross profit = sales – COGS if we have those columns. Or create an index for month number or year for potential seasonal analysis.
This step is about getting the data ready for analysis and model building. Print out a summary (e.g., first few rows, some summary stats) to verify everything loaded correctly. This code will be part of the final CLI tool (likely in an initialization or data module).

Exploratory Data Analysis (EDA): Before jumping into modeling, analyze the historical data to extract insights and inform our modeling approach. This can be done in a Jupyter Notebook or through plots/summary stats printed via the code:

Plot the time series of monthly sales over the years. Observe the trend: Is it upward (growth)? Are there clear seasonal patterns (e.g., higher sales every December or around certain months)? For example, Elips might see spikes in Q4 if institutions rush to use budgets by year-end. Mark any such patterns.

Calculate year-over-year growth rates, seasonal indices (like average sales by month to see which months are above/below average).

Check the volatility of sales – high variance might suggest the need for a robust model or maybe splitting by product type (if one product’s sales are very erratic, maybe forecast it separately).

Analyze expenses relative to sales: e.g., compute historical gross margin percentages per period. If it’s relatively stable (e.g., always ~40% of revenue), a simple assumption might hold; if it fluctuates (perhaps due to currency changes making imports costly at times), note that as something to incorporate into scenarios.

Look at external data correlations: If we have exchange rate data, plot sales or costs against it over time. Perhaps a devaluation leads to higher costs and thus lower margins in subsequent months. This EDA might reveal, for instance, that when TRY had a big drop, profits dipped accordingly.

Summarize key findings: for example, “Sales have grown ~15% annually with a noticeable dip in mid-2020 (possibly pandemic impact) and strong seasonality (peak in Dec, low in Feb). Gross margins average 35%, but dropped to 30% during high exchange-rate volatility.” These insights will guide the forecasting model (e.g., include seasonality) and scenario planning (e.g., include a scenario for currency shock).

(In the CLI context, EDA results can be printed or saved as charts. While not all end-users need to see EDA, performing it ensures the developer understands the data. For the master’s project report, these visualizations and stats would be included to demonstrate understanding of the dataset.)

Develop the Forecasting Model: This is a crucial step – building the model that forecasts future financial metrics (primarily sales revenue, possibly volume or other KPIs). Based on the EDA, choose an appropriate modeling technique:

Time Series Model: Given that we have a time series of sales, an ARIMA model is a logical choice if the data shows autocorrelation and seasonality. We will likely use an ARIMA or SARIMA (seasonal ARIMA) to capture both trend and seasonal patterns. For example, if monthly data shows a yearly cycle, use SARIMA with a 12-month seasonal period. We’ll differentiate the series if needed to make it stationary (ARIMA will handle an integrated component for trend).

We will use Python’s statsmodels.tsa module or pmdarima.auto_arima to fit the model. The procedure: determine orders (p,d,q) for ARIMA by examining autocorrelation plots (ACF/PACF) or let auto_arima find the best parameters. Also determine seasonal order (P,D,Q, s) if using seasonal ARIMA.

Alternative Models: We might compare the ARIMA forecast with a simpler exponential smoothing (ETS) model (using statsmodels ExponentialSmoothing) or with Prophet which is easier for including seasonality and holiday effects (though in B2B sales, holidays might not be as key, except maybe year-end). We could even attempt a basic machine learning approach: e.g., use a regression where time index and maybe external variables predict sales. However, with limited data, ARIMA/SARIMA is usually sufficient and more transparent.

Train the Model: Using historical data up to the most recent completed period, train (fit) the model. This will yield a fitted model we can use for forecasting. Ensure to hold out the last few data points for testing (for example, train on data up to Dec 2023, and reserve Jan–Jun 2024 to see how the model would have predicted those).

Validate the Model: After training, produce forecasts for the hold-out period and compare with actuals. Calculate error metrics (MAE, MAPE, etc.). If errors are large or patterns in residuals exist (maybe the model consistently underestimates peaks), refine the model:

Possibly incorporate a seasonal dummy or additional regressors if using ARIMAX (for instance, include exchange rate as an exogenous variable if that seems to drive part of the variance in costs or if sales volume is sensitive to economic conditions).

Adjust parameters or try a different approach (if ARIMA isn’t capturing a non-linear trend, Prophet or even a small neural network could be tried, but recall the instruction: no overly complex system needed. We should keep the modeling interpretable and reasonably simple).

Once a satisfactory model is achieved (one that captures the trend and seasonality well and has acceptable forecast accuracy), proceed. Document the model choice and its parameters for the final report. For example, “Chosen model: SARIMA(1,1,1)(0,1,1,12) – this captures an annual seasonal effect and provided the lowest AIC and a MAPE of 8% on the validation set.”

Forecast Future Periods: Using the finalized model, generate forecasts for the desired future horizon. In a financial planning context, companies often forecast 12 months ahead (or 4 quarters ahead) on a rolling basis. We will, for example, forecast the next 12 months of sales revenue. The output will be a series of predicted values for each period. If using ARIMA/SARIMA, this is straightforward (e.g. results.forecast(steps=12) gives 12 predictions). If using Prophet, we’d create a future dataframe of dates and use model.predict(future_dates).
Along with point forecasts, it’s good to obtain prediction intervals (confidence ranges), which statsmodels can provide (e.g. 80% or 95% confidence interval for each forecast point). This will be useful when integrating into the financial plan to indicate uncertainty (for instance, best vs worst case outcomes if things vary).
At this stage, we have a projection of sales. If we plan to forecast other elements (like expenses or cash flow), we have options:

Expense Forecasting: We can forecast major expense lines similarly if data permits. For instance, if operating expenses have an increasing trend, one could fit a simple time series or just use an average growth rate assumption. Alternatively, some expenses can be modeled as percentage of sales (variable costs) vs fixed costs. For simplicity, we might assume COGS is a fixed percentage of sales (based on historical average margin). So forecasted COGS = that % * forecasted sales. For fixed operating costs, assume they grow at, say, inflation (~ say 20% in Turkey or adjust per scenario). These assumptions can be refined in scenarios rather than formal models since priority is revenue forecasting.

Assemble Baseline Forecast: Combine the pieces into a baseline financial forecast. E.g., create a table with columns: Month, Forecasted Sales, Forecasted COGS, Forecasted Gross Profit, Forecasted Opex, Forecasted Net Profit. The sales comes from our model, COGS from percentage assumption, Opex from either a small model or constant plus inflation. This baseline will serve as the starting point for planning.

Integrate Scenario Analysis: Now that we have a baseline forecast, enhance the system with scenario analysis capabilities. Scenario analysis allows testing different assumptions easily, which is a key part of financial planning (and is something AI tools excel at, by doing instant “what-if” analysis
rtslabs.com
). We will implement it as follows:

Identify Key Variables for Scenarios: Determine which factors the user might want to tweak. Likely candidates: Sales growth rate (or a multiplier on the forecast, e.g., what if sales are 10% higher/lower than the model predicts), Exchange rate impact (what if the TRY depreciates or appreciates, affecting COGS), Expense adjustments (what if the company hires more staff, increasing fixed costs, or conversely cuts costs).

Define a Few Standard Scenarios: For convenience, define e.g. Base Case (the model’s own forecast and normal assumptions), Best Case (higher sales, favorable FX, etc.), Worst Case (lower sales, or a currency shock causing costs to spike). For example:

Base Case: Use the default forecast. Assume exchange rate stays stable (or whatever baseline we used), no big changes.

High Growth Scenario: Assume demand is higher – e.g. sales +15% over baseline (this could simulate winning extra contracts or market growth). Implement by multiplying the forecast values by 1.15 or adding a growth factor before forecasting (or simply post-process the forecast).

FX Shock Scenario: Assume the lira weakens significantly (e.g., +20% cost on imported goods). Implement by increasing COGS percentage for future periods or adding a one-time cost hit. This will show lower margins in the plan.

Cost Saving Scenario: Assume company undertakes cost-cutting, reducing operating expenses by say 10%. Reflect that in the forecast by adjusting the expense line.

Interactive Input: In the CLI, the user can select a scenario or input custom parameters. For example, we could allow commands like --scenario best or prompt: “Enter custom scenario: expected sales change (%), expected FX rate change (%), etc.” The program will then adjust the forecast accordingly. If a user enters custom values, we apply those adjustments on top of the base forecast. For instance, if user says “sales +5%, TRY falls from 27/USD to 30/USD”, we know to increase sales numbers by 5% and increase COGS proportionally to the currency change.

Recalculate Financials: After adjustments, the module recalculates the financial projection. Because our planning logic is encapsulated (we have formulas for COGS, Opex, etc.), it’s straightforward to plug in new assumptions and get a revised profit forecast. We should output the results clearly, e.g., “Scenario: Best Case – Next year projected revenue = X TL, net profit = Y TL (Z% higher than base case).” Possibly show a month-by-month breakdown for completeness.

Multiple Scenario Output: Optionally, the system can generate a comparison of scenarios side by side. This could be a nice addition: e.g., output a small table or CSV where each column is a scenario and each row quarter’s profit. This would allow Elips Medikal’s team to see the range of outcomes. Modern FP&A often compares best, worst, most-likely outcomes
rtslabs.com
; we emulate that here in a simple way.

Build the Command-Line Interface: Now integrate all the functionality into a cohesive Python CLI application. This involves writing a main driver script (e.g., financial_planner.py) that ties everything together:

Use Python’s argparse to define command-line arguments. For example, --forecast_periods to specify how many future periods to predict, --scenario to pick a scenario name, or --output_file to specify an output report file. Provide help text for each. This allows a user to run the program in one go with desired options (e.g., python financial_planner.py --forecast_periods 12 --scenario best --output_file plan.csv).

Alternatively or additionally, implement an interactive mode if no arguments are given: the program can print a menu and ask the user step by step (some users might prefer a guided approach).

Within the script, structure the code into functions or classes for clarity: e.g., a function load_data() to perform Step 4 (data import), a function train_model() to perform Step 6 and return a model, a function forecast() to generate forecasts, and a function apply_scenario() for Step 8 adjustments. This modular approach makes the code easier to test and modify.

Sequence of execution in main:

Load and prep data.

(Optionally) perform EDA (could be behind a verbosity flag or separate sub-command, since in routine use we might skip plots).

Train or load the forecasting model. (We could save a trained model to disk for reuse to avoid retraining every run, but given data size it’s probably fine to retrain quickly each run. However, if this were used often, saving the model state after first training might be an enhancement.)

Generate baseline forecast for specified periods.

Apply scenario adjustments if any.

Compute the final financial projections.

Output the results: print summary to console, and if --output_file specified, save detailed results to CSV/Excel for further analysis. Possibly also save any charts (like a plot of forecast vs history) as an image file for reporting.

Ensure the CLI handles errors gracefully (e.g., if data file not found, or if the user inputs an unknown scenario name, print a helpful message). Since this is a “system” to be used, user-friendliness counts.

Python Code Emphasis: The implementation will be heavily Python-focused. We will make use of efficient data science libraries to keep code concise. For example, using pandas DataFrames for calculations rather than manual loops. The CLI nature means everything is triggered via code execution rather than manual spreadsheet edits, aligning with the project’s data science/programmatic orientation.

Testing the System with Sample Data: After building the CLI tool, thoroughly test it using the sample data and various scenarios to ensure it works as expected. This step is essential to ensure the “step-by-step” execution is correct and that the system is reliable:

Run the program in base case with a known small dataset where you can manually verify results. For instance, create a very simple dataset (like a linear growth trend) and see if the forecast extrapolates roughly correctly. Verify that scenario adjustments do what they should (e.g., a +10% sales scenario indeed yields 10% higher forecast values).

Test edge cases: What if the user asks for 0 periods forecast (should probably handle or default to some number)? What if an input CSV has a missing month at the end? Make sure the code doesn’t break.

Check the outputs: Open the CSV or printed output and see if the numbers make sense (no negative sales unless we intended to allow that, etc.). Compare the scenario outputs to ensure consistency (worst-case should indeed have lower profits than base, etc., given how we defined them).

If possible, have someone else (or another developer/analyst) run the tool following the instructions, to see if the usage is intuitive and the documentation is sufficient. Since the project could be reviewed by academic supervisors or used by company staff, clarity in how to operate it is important.

Iterate and fix any issues found during testing. For example, if the ARIMA model occasionally fails to converge for certain data, handle that (maybe by providing an initial parameter or trying a different approach in that case). Or if the scenario inputs allow nonsensical values (like a 500% increase), perhaps clamp or warn, depending on need.

Documentation and User Guide: Prepare comprehensive documentation for the system. Even though this is a CLI tool, we need to ensure that users (and evaluators, like professors or the company’s team) can understand and trust it:

Write a README file (or a section in the report) describing the project, setup instructions, and how to run the tool. Include examples of command usage. For instance: “Run python financial_planner.py --forecast_periods 12 --scenario base to generate a one-year baseline forecast. The output will appear in the console and be saved to output_plan.csv.”

Document the data format expected. For example, specify the required columns in the input CSVs (Date and Sales at minimum, plus others if applicable). Since we provide sample data, explain how one could update it with real data from the company in the future (i.e., “replace sales_history.csv with actual data, ensuring the same format”).

Explain the modeling approach briefly: what model was used, and any assumptions in the financial calculations (like “assuming constant 65% gross margin unless scenario adjusted”). This transparency helps build confidence in the projections and allows future improvements.

If applicable, include charts or outputs in the documentation to illustrate what the system produces (e.g., a graph of historical vs forecast sales). While the CLI won’t show the graph, the user can open the saved image. A visual can be very compelling to include in a master’s project report.

Provide guidelines for maintenance: e.g., “Update the model with new data each quarter for better accuracy. Re-train the model yearly or if significant market changes occur. Monitor forecast errors and adjust assumptions as needed.” This shows that the system is not a static one-off, but a living tool that can evolve (which often interests graduate committees, showing awareness of model lifecycle).

Finally, highlight how this system addresses the needs: perhaps write a short section in documentation on “Business Impact”, reiterating how Elips Medikal can use it in practice (quarterly budget meetings, risk management, etc.), and how it demonstrates modern data-driven FP&A aligning with best practices (reducing manual work and providing deeper insights
rtslabs.com
).

Deployment and Future Enhancements: In the final step, consider how to deploy and improve the system (this ensures the project doesn’t end at just a prototype, but shows foresight):

Deployment: Since it’s a simple Python script, deployment might be as easy as placing it on a company computer or server with Python installed. If the finance team is not familiar with command-line, provide a batch file or wrapper that they can double-click to run the analysis. Alternatively, schedule the script to run monthly and email out the report – these are possible extensions to make it more user-friendly in a real business setting.

Security and Data Updates: Make sure any sensitive data is stored securely (if we had actual data). Since we are using dummy data, point out that in real use, one must secure financial data files (perhaps integrate with their database or at least ensure files are not publicly accessible). Also plan for updating the data – e.g., at month-end, someone needs to add the latest figures to the CSV or the script could be extended to append new data automatically if connected to a data source.

Future Improvements: Outline a few ideas for future refinement (to show awareness but without implementing them now, keeping within scope). Examples:

Incorporating more advanced machine learning models (like XGBoost or LSTM neural networks) if the data volume grows, to potentially capture complex patterns. Currently, our solution uses classical time-series modeling; in a big-data scenario, ML could be beneficial.

Adding a dashboard or visualization layer in the future (perhaps using a library like Plotly Dash or a simple web interface) for the finance team to interact with results more easily. This wasn’t required now, but it’s a logical next step if adoption is successful.

Extending the scope beyond just sales & profit forecasting – maybe integrate inventory level predictions, or optimize ordering (since Elips likely has to order equipment from abroad, aligning inventory with forecasted demand could reduce holding costs – a nice operations research addition). Another extension could be customer-level analysis (e.g., using data science to identify which client segments drive growth and focusing sales efforts accordingly).

Automating scenario suggestions using AI: e.g., the system could analyze macroeconomic data and automatically warn “high inflation scenario likely” or something – this is advanced but shows how data science can continuously enhance financial planning.

Finalize Deployment: Once ready, deliver the system to the intended user. For the scope of our project, this means having the code, documentation, and example data available. In a real scenario, it would involve running a demo for the company’s team and training them briefly on how to use the tool. Since we kept it as a CLI, training is minimal (just follow the README steps). The end result is that Elips Medikal’s finance team has a working, data-driven planning system at their disposal.

Conclusion and Project Outlook

By completing all the above steps, we will have developed a fully functional Financial Planning System tailored for Elips Medikal. This system harnesses the power of data science in a finance context: it analyzes historical performance, generates accurate forecasts, and allows interactive scenario planning to guide strategic decisions. The final deliverable is a Python CLI program that is both practical for the company and intellectually rich for academic purposes. It transforms a once manual, spreadsheet-driven process into an automated, reproducible workflow that can be run anytime new data arrives or when management needs to evaluate options.

This project demonstrates how even a mid-sized distributor like Elips Medikal can leverage modern analytics to plan smarter with limited resources – aligning with trends where even lean organizations use data-driven “what-if” scenario modeling instead of static forecasts
rtslabs.com
. The implementation was kept feasible and not overly complex, focusing on core achievable features: time-series forecasting and basic scenario analysis, implemented with accessible Python libraries. Yet, it remains extensible for future enhancements.

In summary, we now have a robust, data-driven financial planning tool that will help Elips Medikal anticipate the future, allocate resources efficiently, and maintain resilience against uncertainties (like currency swings or market changes). The project not only provides immediate business value (better financial visibility, time savings in analysis, deeper insights)
rtslabs.com
rtslabs.com
 but also serves as an excellent showcase of integrating data science with finance – exactly the interdisciplinary innovation that attracts modern master’s programs and forward-thinking organizations.

Sources: The plan was informed by research on Elips Medikal’s business model and current best practices in data-driven financial planning. Key references include the company’s profile and product scope
emis.com
elipsltd.com.tr
, and expert insights on applying AI/analytics to financial planning and scenario analysis in real businesses
rtslabs.com
rtslabs.com
. All these guided the creation of a tailored, effective solution for the company. With this system ready to execute step by step, Elips Medikal can confidently move toward a more data-informed financial strategy.