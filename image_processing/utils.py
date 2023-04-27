def set_barplot_configs(fig):
    fig.update_layout(yaxis_range=[0, 1])
    fig.update_layout(
        font_size=25,
        legend_font_size=20,
        title_font_size=100,
    )

    fig['layout']['xaxis']['title'] = 'Category'
    fig['layout']['yaxis']['title'] = 'Probability (%)'

    fig['layout']['xaxis']['titlefont']['size'] = 25
    fig['layout']['yaxis']['titlefont']['size'] = 25