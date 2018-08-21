procedure Test is
   P : access procedure;
begin
   if P = null then
      P := Test'Access;
   end if;
end Test;
