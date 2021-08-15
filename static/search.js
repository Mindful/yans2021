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
            zeroline: false
        },
        yaxis: {
            type: 'linear',
            zeroline: false
        },
        zaxis: {
            type: 'linear',
            zeroline: false
        }
    },
    title: '3d point clustering',
};

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
            size: cluster['is_user_input'] ? 60 : 20
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

function draw_visualization(json_data) {
    console.log(json_data)
    let data = json_data['clusters'].map(to_data)
    Plotly.react('visualization', data, visualization_layout);
}

document.addEventListener("DOMContentLoaded", () => {
    let search_form = document.getElementById('search_form')
    let search_text_area = document.getElementById('search_text_area')


    search_form.addEventListener('submit', function (e) {
        if (validate_search_content(search_text_area.value)) {
            fetch('/search?text='+search_text_area.value)
                .then(response => response.json())
                .then(data => draw_visualization(data))
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