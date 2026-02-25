import plotly.express as px

plotly_template = 'plotly_white'

def plot_grid(X, title=None):
    fig = px.imshow(X, 
        color_continuous_scale=px.colors.qualitative.Light24,
        x = X.columns.to_list(),
        y = X.index.to_list(),
        text_auto=False,
        title=title, 
        template=plotly_template)
    fig.update_traces(hoverongaps=False)
    fig.update_coloraxes(showscale=False)
    fig.show()


def plot_map(X, title=None):
    axis_labels = [':'.join(map(str,x)) for x in X.index.values]
    # axis_labels = [str(x[0]).zfill(2) + ':' + str(x[1]).zfill(2) for x in X.index.values]
    cmap = px.colors.diverging.Spectral
    # cmap = px.colors.sequential.Blues
    fig = px.imshow(X.values, x = axis_labels, y = axis_labels,
        height=1000, width=1200,
        color_continuous_scale=cmap,
        title=title, 
        template=plotly_template)
    fig.show()


class CorrelationHeatMap:

    def __init__(self, X, title=None):
        self.X = X
        self.title = title
        self.axis_labels = [':'.join(map(str,x)) for x in X.index.values]
        self.cmap = px.colors.diverging.Spectral
        self.height = 1000
        self.width = 1200
        self.plotly_template = "plotly_white"

    def plot(self):
        fig = px.imshow(self.X.values, 
            x = self.axis_labels, y = self.axis_labels,
            height=self.height, width=self.width,
            color_continuous_scale=self.cmap,
            title=self.title, 
            template=self.plotly_template)
        fig.show()
        
        

    