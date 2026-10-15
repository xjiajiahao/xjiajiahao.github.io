/**
 * 站点部署在子路径时（例如 GitHub Pages 项目页 /xiebro-blog/），
 * 站内绝对链接必须带上 base 前缀，否则会指向域名根目录而 404。
 */
export const BASE = import.meta.env.BASE_URL;

/** 给站内绝对路径加上 base 前缀；外部链接与锚点原样返回。 */
export function withBase(path: string): string {
  if (/^(?:[a-z][a-z0-9+.-]*:)?\/\//i.test(path) || path.startsWith("#")) {
    return path;
  }
  return `${BASE.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

/** 判断某个导航项是否为当前页面，用于 aria-current。 */
export function isCurrent(pathname: string, href: string): boolean {
  const target = withBase(href);
  return href === "/" ? pathname === target : pathname.startsWith(target);
}
