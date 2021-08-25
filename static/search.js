const visualization_layout = {
    autosize: true,
    height: 1200,
    width: 1200,
    scene: {
        aspectratio: {
            x: 1,
            y: 1,
            z: 1
        },
        camera: {
            center: {
                x: 0,
                y: 0,
                z: 0
            },
            eye: {
                x: 1.25,
                y: 1.25,
                z: 1.25
            },
            up: {
                x: 0,
                y: 0,
                z: 1
            }
        },
        xaxis: {
            type: 'linear',
            zeroline: false,
            showspikes: false,
        },
        yaxis: {
            type: 'linear',
            zeroline: false,
            showspikes: false,
        },
        zaxis: {
            type: 'linear',
            zeroline: false,
            showspikes: false,
        }
    },
    title: '3d point clustering',
};

const visualization_options = {
    showTips: false,
    modeBarButtons: [
    [{
            name: 'Up one level',
            icon: Plotly.Icons.camera,
            click: () => {
                let parent_tree = search_state.search_data.tree.split('-')
                parent_tree.pop()
                query_cluster(parent_tree.join('-'))
            }
        }
    ]
]
}

let search_state;

function unpack(rows, key) {
    return rows.map(function (row) {
        return row[key];
    });
}

function to_data(cluster) {
    return {
        x: unpack(cluster['data'], 'x'),
        y: unpack(cluster['data'], 'y'),
        z: unpack(cluster['data'], 'z'),
        text: unpack(cluster['data'], 'text'),
        mode: 'markers',
        type: 'scatter3d',
        name: cluster['name'],
        hovertemplate: '%{text}',
        marker: {
            color: cluster['color'],
            size: cluster['is_user_input'] ? 60 : 20,
            line: {
                color: 'rgb(231, 99, 250)',
                width: cluster['is_user_input'] ? 6 : 0
            }
        }
    }
}

function validate_search_content (content) {
    let lbr_count = content.split('[').length - 1
    let rbr_count = content.split(']').length - 1

    if (lbr_count !== 1 || rbr_count !== 1) {
        return false
    }

    let first_lbr_index = content.indexOf('[')
    let first_rbr_index = content.indexOf(']')

    if (first_lbr_index >= first_rbr_index) {
        return false
    }

    return true
}

let visualization_initialized = false;

function query_cluster(tree) {
    let new_query_data = { ...search_state.search_data, tree: tree}
    console.log(new_query_data)
    fetch('/subcluster', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(new_query_data)
    }).then(response => response.json()).then(data => draw_visualization(data))
}

function draw_visualization(json_data) {
    search_state = json_data
    console.log(json_data)
    let data = json_data['clusters'].map(to_data)
    let vis_plot = document.getElementById('visualization')
    Plotly.react('visualization', data, visualization_layout, visualization_options);

    if (!visualization_initialized) {
        vis_plot.on('plotly_legenddoubleclick', (event) => {
            query_cluster(search_state.search_data.tree + "-" + event.curveNumber)
            return false;
        })
        visualization_initialized = true;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    let search_form = document.getElementById('search_form')
    let search_text_area = document.getElementById('search_text_area')


    search_form.addEventListener('submit', function (e) {
        if (validate_search_content(search_text_area.value)) {
            fetch('/search?', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({'text': search_text_area.value})
            }).then(response => response.json()).then(data => draw_visualization(data))
        }
        e.preventDefault()
        e.stopPropagation()
    })


    search_text_area.addEventListener('change', function () {
        if (validate_search_content(search_text_area.value)) {
            search_text_area.classList.remove('is-invalid')
        } else {
            search_text_area.classList.add('is-invalid')
        }
    })
})