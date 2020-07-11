<ul class="postlist">
  {% assign readings = site.collections | where: "type", "reading" %}
  {% for collection in readings %}
  <h3>{{ collection.docs[0].tags }}</h3>
  {% for post in collection.docs limit:5 %}
    <li>
      <b>{{ post.date | date: "%Y-%m-%d" }}</b>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <p>{{ post.excerpt | strip_html | truncate: 200 }}</p>
    </li>
  {% endfor %}
  {% endfor %}
</ul>

