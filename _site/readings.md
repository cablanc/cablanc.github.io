<ul>
  {% for post in site.gnn %}
    <li>
      <a href="{{ post.url }}"/a>
      {{ post.content }}
    </li>
  {% endfor %}
</ul>