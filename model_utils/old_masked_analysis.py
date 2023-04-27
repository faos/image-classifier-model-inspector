"""st.header('Masked Analysis')
st.markdown(""<hr style="width:100%; text-align:left; margin:0">"", unsafe_allow_html=True)


stroke_width = 1
stroke_color = "#ffffff"

bg_color = "#000000"
img_size = 224
row0_spacer1, row0_2, row0_spacer3, row0_3, row0_spacer4 = st.columns((.1, 2.0, .05, 2.0, 0.1))

with row0_2:
    # Specify canvas parameters in application
    drawing_mode = st.selectbox(
        "Drawing tool:", ("rect", "circle", "polygon")
    )

with row0_3:
    # Specify canvas parameters in application
    draw_operation = st.selectbox(
        "Drawing operation:", ("Erase", "Selection")
    )

realtime_update = False

if bg_image:
    print('\n\n\n')
    print('bg_image:', bg_image)
    img = Image.open(bg_image).resize((img_size, img_size))
    vec = np.asarray(img)
    height, width, channels = vec.shape
    print(vec.shape)
    print('stroke_width', stroke_width)
    print('stroke_color', stroke_color)

    row01_spacer1, row01_1, row0_spacer2, row01_2, row01_spacer3, row01_3, row01_spacer4 = st.columns((.1, 2.0, .1, 2.0, .1, 2.0, 0.1))

    with row01_1:
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.3)",  # Fixed fill color with some opacity
            #fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=img,
            update_streamlit=realtime_update,
            height=img.size[1],
            width=img.size[0],
            drawing_mode=drawing_mode,
            point_display_radius=0,
            key="canvas",
        )

        #logits = inference.Predictor()(st.session_state["model"], img)
        #df_raw = build_dataframe(logits)
        #fig = px.bar(df_raw, x='categories', y='probabilities')
        #print("I N F E R E N C E - RAW:", logits)
        #st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with row01_2:
        # Do something interesting with the image data and paths
        if canvas_result.image_data is not None:
            print("SUMMARY", type(canvas_result.image_data[:,:,1:]), canvas_result.image_data.shape)
            #st.image(canvas_result.image_data)

            mask = Image.fromarray(canvas_result.image_data[:,:,1:])
            
            #st.image(mask)

            mask = mask.resize((width, height))
            mask = np.asarray(mask)
            
            #mask.flags['WRITEABLE'] = True
            #mask[mask.copy() > 0] = 1

            print("MASK SUM:", (mask > 0).sum())
            print("ERASE MASK SUM:", (1 - (mask > 0)).sum(), (1 - (mask > 0)).shape, ((1 - (mask > 0)) < 0).sum())
            print('MASK:', mask.shape)

            if draw_operation == 'Erase':
                #img_ans = Image.fromarray((np.ones(mask.shape) - (mask > 0)) * vec, mode="RGB")
                img_ans = Image.fromarray((1 - (mask > 0)).astype(np.uint8) * vec, mode="RGB")
            
            elif draw_operation == 'Selection':
                img_ans = Image.fromarray((mask > 0) * vec, mode="RGB")

            st.image(img_ans.resize(img.size), use_column_width=True)

            #logits = inference.Predictor()(st.session_state["model"], img_ans)
            #df_texture = build_dataframe(logits)
            #fig = px.bar(df_texture, x='categories', y='probabilities')
            #print("I N F E R E N C E - TEXTURE:", df_texture)
            #st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    with row01_3:
        logits_raw = inference.Predictor()(st.session_state["model"], img)
        logits_txt = inference.Predictor()(st.session_state["model"], img_ans)

        df = build_dataframe_pair(logits_raw, logits_txt)

        fig = px.bar(df, x='categories', y='probabilities', color="type", barmode="group")
        print("I N F E R E N C E - TEXTURE:", df)
        fig.update_layout(yaxis_range=[0, 1], margin= dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        #logits_raw = inference.Predictor()(st.session_state["model"], img)
        #logits_txt = inference.Predictor()(st.session_state["model"], img_ans)

        #df = build_dataframe_pair(logits_raw, logits_txt)

        #fig = px.bar(df, x='categories', y='probabilities', color="type", barmode="group")
        #print("I N F E R E N C E - TEXTURE:", df)
        #fig.update_layout(yaxis_range=[0, 1], margin= dict(l=0,r=0,b=0,t=0))
        #st.plotly_chart(fig, theme="streamlit", use_container_width=True)"""