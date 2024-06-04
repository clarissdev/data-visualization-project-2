library(shiny)
library(DT)
library(ggplot2)
library(dplyr)
library(plotly)
library(GGally)
library(psych)

ui <- navbarPage(
    "Machine Learning Pipeline",
    tabPanel(
        "About",
        fluidPage(
            titlePanel(p("About the Project", style = "color:#3474A7")),
            h4("High-Level Goal"),
            p("Our objective is to make machine learning more accessible and inclusive. By creating an intuitive simulation of a machine learning pipeline, we aim to enhance the learning experience, allowing users to easily grasp complex concepts and apply machine learning techniques in a practical, hands-on manner."),
            br(), br(),
            h4("Project Description"),
            p("This project offers a user-friendly simulation of a machine learning pipeline using R Shiny. Through interactive modules, users can upload datasets, preprocess data, select and train machine learning algorithms, and evaluate model performance. Our platform provides an engaging, hands-on learning experience, making machine learning concepts accessible to a broad audience."),
            br(), br(),
            h4("Steps to complete and evaluate a machine learning model"),
            tags$ol(
                tags$li("Upload Data: Choose the data you want to use"),
                tags$li("Data Summary: Give an overview by providing a data summary"),
                tags$li("Data Exploration: Preprocess data by deleting/filling NAs and creating simple plots"),
                tags$li("Build Model: Pick variables, split train/test set, choose algorithms"),
                tags$li("Evaluation: Evaluate the model performance")
            ),
            br(), br()
        )
    ),
    tabPanel(
        "1. Upload",
        fluidPage(
            titlePanel(p("Upload the Data")),
            sidebarLayout(
                sidebarPanel(
                    fileInput(
                        inputId = "upload",
                        label = "Upload data (please upload csv.)",
                        accept = c(".csv")
                    ),
                    br(),
                    uiOutput("target")
                ),
                mainPanel(
                    h4("Data Uploaded:"),
                    tableOutput("files"),
                    br(), br(),
                    h4("Data Preview:"),
                    DTOutput("head")
                )
            )
        )
    ),
    tabPanel(
        "2. Data Summary",
        titlePanel(p("Data Summary")),
        sidebarLayout(
            sidebarPanel(
                p("This section provides a summary of the uploaded data."),
                selectInput(inputId = "variable", label = "Select variable for violin plot:", choices = NULL)
            ),
            mainPanel(
                h4("Summary of Data"),
                tableOutput("dataSummary"),
                br(), br(),
                h4("Violin Plot"),
                plotlyOutput("violinPlot")
            )
        )
    ),
    tabPanel(
        "3. Data Exploration",
        fluidPage(
            titlePanel(p("Data Preprocessing")),
            sidebarLayout(
                sidebarPanel(
                    h4("Data Preprocessing"),
                    selectInput("na_action", "Choose NA handling method:",
                        choices = c("None", "Delete all NAs", "Fill NAs with mean")
                    ),
                    actionButton("apply_preprocess", "Apply"),
                    br(), br(),
                    h4("Data Plotting"),
                    selectInput("singlePlotGeom", "Select plot type",
                        choices = c("point", "boxplot", "histogram", "density"),
                        selected = "histogram"
                    ),
                    uiOutput("expXaxisVarSelector"),
                    uiOutput("expYaxisVarSelector"),
                    uiOutput("expColorVarSelector"),
                    br(), br(),
                    h4("3D Plot Selection"), # New section for 3D plot selection
                    uiOutput("exp3DXaxisVarSelector"),
                    uiOutput("exp3DYaxisVarSelector"),
                    uiOutput("exp3DZaxisVarSelector"),
                    uiOutput("exp3DColorVarSelector")
                ),
                mainPanel(
                    h4("One and Two Variable Plot"),
                    plotlyOutput("expSinglePlot"),
                    h4("Pairs Plot (only non-zero variance variables shown)"),
                    plotlyOutput("expPairsPlot", width = "100%", height = "800px"),
                    h4("3D Scatter Plot"), # New 3D scatter plot
                    plotlyOutput("exp3DPlot")
                )
            )
        )
    )
)

server <- function(input, output, session) {
    data <- reactiveVal(NULL)
    processed_data <- reactiveVal(NULL)

    observeEvent(input$upload, {
        req(input$upload)
        df <- read.csv(input$upload$datapath)
        data(df)
        processed_data(df)

        output$files <- renderTable({
            data.frame(Name = input$upload$name, Size = input$upload$size)
        })

        output$head <- DT::renderDT({
            datatable(data(), filter = "top", editable = TRUE, rownames = FALSE, options = list(
                dom = "lrtip",
                lengthMenu = list(c(10, 50, 100, -1), c(10, 50, 100, "All"))
            ))
        })

        updateSelectInput(session, "variable", choices = names(df))
        updateSelectInput(session, "expXaxisVarSelector", choices = names(df))
        updateSelectInput(session, "expYaxisVarSelector", choices = c("None", names(df)), selected = "None")
        updateSelectInput(session, "expColorVarSelector", choices = c("None", names(df)))
        updateSelectInput(session, "exp3DXaxisVarSelector", choices = names(df)) # Update for 3D X-axis variable selector
        updateSelectInput(session, "exp3DYaxisVarSelector", choices = names(df)) # Update for 3D Y-axis variable selector
        updateSelectInput(session, "exp3DZaxisVarSelector", choices = names(df)) # Update for 3D Z-axis variable selector
        updateSelectInput(session, "exp3DColorVarSelector", choices = c("None", names(df))) # Update for 3D Color variable selector
    })

    output$dataSummary <- renderTable(
        {
            df <- data()
            if (is.null(df)) {
                return(NULL)
            }
            summary_df <- describe(df[, !(names(df) %in% c(input$variable))])
            summary_df
        },
        rownames = TRUE
    )

    output$violinPlot <- renderPlotly({
        df <- data()
        var <- input$variable
        if (is.null(df) || is.null(var)) {
            return(NULL)
        }

        p <- ggplot(df, aes_string(x = var, y = var)) +
            geom_violin() +
            labs(title = paste("Violin Plot of", var), x = var, y = var) +
            theme_minimal()

        ggplotly(p)
    })

    observeEvent(input$apply_preprocess, {
        df <- data()
        if (is.null(df)) {
            return(NULL)
        }

        if (input$na_action == "Delete all NAs") {
            df <- na.omit(df)
        } else if (input$na_action == "Fill NAs with mean") {
            df <- df %>% mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))
        }
        processed_data(df)
    })

    output$expXaxisVarSelector <- renderUI({
        df <- processed_data()
        if (is.null(df)) {
            return(NULL)
        }
        selectInput("expXaxisVar", "X-axis variable", choices = names(df))
    })

    output$expYaxisVarSelector <- renderUI({
        df <- processed_data()
        if (is.null(df)) {
            return(NULL)
        }
        selectInput("expYaxisVar", "Y-axis variable", choices = c("None", names(df)))
    })

    output$expColorVarSelector <- renderUI({
        df <- processed_data()
        if (is.null(df)) {
            return(NULL)
        }
        selectInput("expColorVar", "Color variable", choices = c("None", names(df)))
    })

    output$exp3DXaxisVarSelector <- renderUI({
        df <- processed_data()
        if (is.null(df)) {
            return(NULL)
        }
        selectInput("exp3DXaxisVar", "3D X-axis variable", choices = names(df))
    })

    output$exp3DYaxisVarSelector <- renderUI({
        df <- processed_data()
        if (is.null(df)) {
            return(NULL)
        }
        selectInput("exp3DYaxisVar", "3D Y-axis variable", choices = names(df))
    })

    output$exp3DZaxisVarSelector <- renderUI({
        df <- processed_data()
        if (is.null(df)) {
            return(NULL)
        }
        selectInput("exp3DZaxisVar", "3D Z-axis variable", choices = names(df))
    })

    output$exp3DColorVarSelector <- renderUI({
        df <- processed_data()
        if (is.null(df)) {
            return(NULL)
        }
        selectInput("exp3DColorVar", "3D Color variable", choices = c("None", names(df)))
    })

    output$expSinglePlot <- renderPlotly({
        df <- processed_data()
        if (is.null(df)) {
            return(NULL)
        }

        x_var <- input$expXaxisVar
        y_var <- input$expYaxisVar
        color_var <- input$expColorVar

        p <- ggplot(df, aes_string(x = x_var))

        if (input$singlePlotGeom %in% c("histogram", "density")) {
            geom <- switch(input$singlePlotGeom,
                "histogram" = geom_histogram(),
                "density" = geom_density()
            )

            p <- p + geom
        } else {
            if (y_var != "None") {
                p <- p + aes_string(y = y_var)
            }

            if (color_var != "None") {
                p <- p + aes_string(color = color_var)
            }

            geom <- switch(input$singlePlotGeom,
                "point" = geom_point(),
                "boxplot" = geom_boxplot()
            )

            p <- p + geom
        }

        ggplotly(p + labs(title = paste(input$singlePlotGeom, "of", x_var, if (y_var != "None") paste("and", y_var))) +
            theme_minimal())
    })

    output$expPairsPlot <- renderPlotly({
        df <- processed_data()
        if (is.null(df)) {
            return(NULL)
        }
        df <- df %>% select_if(~ var(.) != 0)

        p <- ggpairs(df) + theme_minimal()
        ggplotly(p)
    })

    output$exp3DPlot <- renderPlotly({
        df <- processed_data()
        if (is.null(df)) {
            return(NULL)
        }

        x_var <- input$exp3DXaxisVar
        y_var <- input$exp3DYaxisVar
        z_var <- input$exp3DZaxisVar
        color_var <- input$exp3DColorVar

        if (is.null(x_var) || is.null(y_var) || is.null(z_var) || x_var == "" || y_var == "" || z_var == "") {
            return(NULL)
        }

        plot_ly(df,
            x = ~ get(x_var), y = ~ get(y_var), z = ~ get(z_var), color = ~ get(color_var), colorscale = "Viridis",
            type = "scatter3d", mode = "markers"
        ) %>%
            layout(scene = list(
                xaxis = list(title = x_var),
                yaxis = list(title = y_var),
                zaxis = list(title = z_var)
            ))
    })
}

shinyApp(ui = ui, server = server)