# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8080/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc 
import pandas as pd
import pandas_datareader.data as web
import datetime

# -------------------------- PYTHON FUNCTIONS ---------------------------- #

def load_data(data_file=None):
        '''0.0.1  Create a function to load the dataset which was extracted from the Data Warehouse'''
        return pd.read_csv(data_file) 

def get_df_country_metrics(dfObj=None, country=None, theme=None):
        '''0.0.3  Create a function which takes the dataset, a country, and a theme and returns only the data for that 
        specific input and writes the metrics/features into columns (from long to wide format)'''
        dfObj = dfObj[ (dfObj['Country'] == country) & (df['Theme'] == theme) ]
        dfObj = dfObj.pivot_table(index='Year', 
                    columns='Metric', 
                    values='Value')#.reset_index(drop=True)
        dfObj.columns = [str(col) for col in dfObj.columns]  
        #dfObj.style.highlight_null('red')          
        #st.dataframe(dfObj.style.applymap(lambda cell: 'color:red' if pd.isnull(cell) else ''))
        print(dfObj.style.applymap(lambda cell: 'color:red' if pd.isnull(cell) else ''))
        return dfObj

def plot_bar(dfObj=None, theme=None, sumcheck_var=None):
    '''Function to plot stacked bar charts, which are used to see if individual metrics components 
    add up to the total amount provided. The total amount is overlaid with a line graph. For example, 
    the sum of female and male injured persons should equal the total amount of injured persons.'''
    print(f"features: {dfObj.columns}")
    sumcheck_dict = sumcheck_var
    flag = 0
    sub_list = sumcheck_dict.get(theme)
    #print(f"Sublist for theme {theme}: {sub_list}")

    
    if sub_list is not None:
        fig = make_subplots(rows=1, cols=len(sumcheck_dict.get(theme)), start_cell="bottom-left")
        for i in range(len(sub_list)):
            sub_list2 = sub_list[i]
            test_list = dfObj.columns 
            #print(f"Testlist: {test_list}")
            #print(f"Sublist: {sub_list2}")
                
            if (set(sub_list2).issubset(set(test_list))):
                flag = 1
                data = []
                for j in range(len(sumcheck_dict.get(theme)[i])-1):
                    #print(f"subplot col {j}")
                    #print(sumcheck_dict.get(theme)[i][j])
                    data.append(go.Bar(name = sumcheck_dict.get(theme)[i][j], x=dfObj.index, y=dfObj[sumcheck_dict.get(theme)[i][j]]))    
                    
                    #for name in range(len(sumcheck_dict.get(theme))):
                    #st.write(data)
                    fig.add_trace(go.Bar(name = sumcheck_dict.get(theme)[i][j], x=dfObj.index, y=dfObj[sumcheck_dict.get(theme)[i][j]]), 
                                    row=1, col=i+1)
                    # Change the bar mode
                    fig.update_layout(barmode='stack') 
                    fig.add_trace(go.Scatter(
                        x = dfObj.index,
                        y = dfObj[sumcheck_dict.get(theme)[i][-1]],
                        name = sumcheck_dict.get(theme)[i][-1],
                        connectgaps=True),
                        row=1,
                        col=i+1
                        )
        return fig                       
    else:
    #print("Nothing to plot. I continue to next subplot.")
    #st.plotly_chart(fig, use_container_width=True)
        print("")
    # printing result
    if (flag) :
        print("")
    else :
        print("")





# -------------------------- LOAD DATA ---------------------------- 

df = load_data(data_file='irfdashboard/data/WRS Data 2000-2019 PIVOT Check version 20210827.csv')        
print(df.sample(n=10))

# -------------------------- DASH ---------------------------- #

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )




# -------------------------- PROJECT DASHBOARD ---------------------------- #

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Jumbotron([html.H1("IRF Data Warehouse Country Explorer", className="display-3"),
                        html.P(
                            "The easy way to see a countries' development over time for several indicators, its comparison to different countries, or income groups, in an interactive App.",
                            className="lead",
                        ),
                        html.P("Your country analysis starts here!", className=" font-italic"),
                        html.Hr(className="my-2"),
                        html.Div(
                            [
                                dbc.Button(
                                    "Theme Explorer",
                                    color="primary",
                                    outline=True,
                                    href="/theme_explorer",
                                    className="mr-2",
                                    size="sm",
                                ),
                                dbc.Button(
                                    "Country Analysis Templates",
                                    color="primary",
                                    outline=True,
                                    href="/dash_bootstrap_templates",
                                    className="mr-2",
                                    size="sm",
                                ),
                                dbc.Button(
                                    "Graph Gallery",
                                    id="graph_gallery",
                                    color="primary",
                                    outline=True,
                                    href="/graph_gallery",
                                    className="mr-2",
                                    size="sm",
                                ),
                                dbc.Button(
                                    "Cheatsheet",
                                    id="cheatsheet",
                                    color="primary",
                                    outline=True,
                                    href="/cheatsheet",
                                    className="mr-2",
                                    size="sm",
                                ),
                            ],
                            className="mt-2",
                        )
                                ]),width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='my-dpdn', multi=False, value='Cameroon', options=[{'label': x, 'value':x} for x in sorted(df["Country"].unique())]),
            dcc.Graph(id='line-fig', figure={})
        ], #width={'size': 6, 'offset': 0, 'order': 1},
            xs=12, sm=12, md=12, lg=5, xl=5
        ),

        dbc.Col([
            dcc.Dropdown(id='my-dpdn2', multi=True, value=['Cameroon', 'Australia'], options=[{'label': x, 'value':x} for x in sorted(df["Country"].unique())]),
            dcc.Graph(id='line-fig2', figure={})
        ], #width={'size': 6, 'offset': 0, 'order': 2},
            xs=12, sm=12, md=12, lg=5, xl=5
        )
    ], no_gutters=False, justify='center'),

    dbc.Row([
        dbc.Col([
            html.P("Select country:", style={'textDecoration': "underline"}),
            dcc.Checklist(id='my-checklist', value=['Benin', 'Burundi', 'Cameroon'], 
                options=[{'label': x, 'value':x} for x in sorted(df["Country"].unique()[0:10])],
                labelClassName='mr-3 text-success'),
            dcc.Graph(id='my-hist', figure={})
            ], #width={'size': 6, 'offset':0, 'order':1},
                xs=12, sm=12, md=12, lg=5, xl=5
        ),

        dbc.Col([
            dbc.Card(
                [
                    dbc.CardBody(
                        html.P(
                            "We're better together. Help each other out!",
                            className="card-text")
                    ),
                    dbc.CardImg(
                        src="https://media.giphy.com/media/Ll0jnPa6IS8eI/giphy.gif",
                        bottom=True),
                ],
                style={"width": "24rem"},
            )
        ], #width={'size': 6, 'offset':0, 'order':2},
            xs=12, sm=12, md=12, lg=5, xl=5
        )
    ], align='center')
], fluid=True)    




# Callback section: connecting the components
# ************************************************************************
# Line chart - Single
@app.callback(
    Output('line-fig', 'figure'),
    Input('my-dpdn', 'value')
)
def update_graph(country_slctd):
    #dff = get_df_country_metrics(dfObj=df, country=country_slctd, theme='Road Traffic')
    dff = df[df['Country'].isin([country_slctd]) & df['Metric'].isin(['Passenger Car Traffic'])]
    figln = px.line(dff, x='Year', y='Value', template='seaborn')
    return figln


# Line chart - multiple
@app.callback(
    Output('line-fig2', 'figure'),
    Input('my-dpdn2', 'value')
)
def update_graph(country_slctd):
    dff = df[df['Country'].isin(country_slctd) & df['Metric'].isin(['Passenger Car Traffic'])]
    figln2 = px.line(dff, x='Year', y='Value', color='Country', template='seaborn')
    return figln2


# Connected Scatterplots #Histogram
@app.callback(
    Output('my-hist', 'figure'),
    Input('my-checklist', 'value')
)
def update_graph(country_slctd):
    dfObj = df[df['Country'].isin(country_slctd)]
    dfObj = dfObj.pivot_table(index=['Country', 'Year'], 
                columns=['Metric'], 
                values='Value').reset_index(drop=False)
    fighist = px.line(dfObj, x='GDP Growth', y='Population', color="Country", text="Year", template='seaborn')
    fighist.update_traces(textposition="bottom right")
    #dff = dff[dff['Year']=='2019']
    #fighist = px.histogram(dff, x='Metric', y='Close')
    return fighist
 



if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)